//! /ainur — parallel multi-agent orchestration.
//!
//! Named after the Ainur from Tolkien's Eä mythology — beings who
//! create through parallel song, each contributing their own theme
//! to the Great Music.
//!
//! Flow:
//!   1. Planner LLM call decomposes the task into N subtasks
//!   2. N worker LLM calls execute in parallel (with tool access)
//!   3. Merger LLM call combines results into a single response

use crate::error::{Error, Result};
use crate::llm::{ContentBlock, LlmProvider, Message, Role, StopReason, ToolDef};
use crate::safety::SafetyLayer;
use crate::tools::ToolRegistry;
use std::sync::Arc;

const PLANNER_PROMPT: &str = "\
You are a task decomposition planner. Given a task and a number N, \
break it into exactly N independent subtasks that can be executed in parallel.

Respond with ONLY a JSON array of N strings, each being a subtask description. \
No explanation, no markdown, just the JSON array.

Example for N=3 and task \"compare cloud providers\":
[\"Research AWS pricing and services for web hosting\", \
\"Research GCP pricing and services for web hosting\", \
\"Research Azure pricing and services for web hosting\"]";

const MERGER_PROMPT: &str = "\
You are a result synthesizer. You will receive results from multiple parallel workers. \
Combine them into a single cohesive response. Be concise and well-structured. \
Use the worker results as-is — do not hallucinate additional information.";

/// Parse `/ainur N <task>` and return (count, task).
pub fn parse_ainur(input: &str) -> Option<(usize, &str)> {
    let rest = input.strip_prefix("/ainur ")?;
    let rest = rest.trim_start();
    // First token is the count
    let space = rest.find(' ')?;
    let count: usize = rest[..space].parse().ok()?;
    if count == 0 || count > 10 {
        return None;
    }
    let task = rest[space..].trim();
    if task.is_empty() {
        return None;
    }
    Some((count, task))
}

/// Status callback type for reporting progress.
pub type OnStatus = Box<dyn Fn(&str) + Send>;

/// Execute the ainur pipeline: plan → parallel workers → merge.
pub async fn execute(
    count: usize,
    task: &str,
    llm: &Arc<dyn LlmProvider>,
    tools: &ToolRegistry,
    safety: &mut SafetyLayer,
    tool_defs: &[ToolDef],
    system_prompt: &str,
    max_turns: usize,
    on_status: OnStatus,
) -> Result<String> {
    // 1. Plan: decompose into subtasks
    on_status(&format!("/ainur {count} — decomposing task..."));

    let subtasks = plan(count, task, llm).await?;

    if subtasks.len() != count {
        return Err(Error::Tool(format!(
            "planner returned {} subtasks, expected {count}",
            subtasks.len()
        )));
    }

    for (i, sub) in subtasks.iter().enumerate() {
        on_status(&format!("♪ Ainur {}/{count}: {sub}", i + 1));
    }

    // 2. Execute workers in parallel
    let worker_futures: Vec<_> = subtasks
        .iter()
        .enumerate()
        .map(|(i, subtask)| {
            let llm = llm.clone();
            let tool_defs = tool_defs.to_vec();
            let system = system_prompt.to_string();
            let subtask = subtask.clone();
            let tools = tools.clone();

            tokio::spawn(async move {
                let result = run_worker(
                    &llm,
                    &subtask,
                    &tools,
                    &tool_defs,
                    &system,
                    max_turns,
                )
                .await;
                (i, result)
            })
        })
        .collect();

    let mut results: Vec<(usize, String)> = Vec::with_capacity(count);
    for handle in worker_futures {
        match handle.await {
            Ok((i, Ok(text))) => results.push((i, text)),
            Ok((i, Err(e))) => results.push((i, format!("Worker error: {e}"))),
            Err(e) => results.push((0, format!("Task join error: {e}"))),
        }
    }
    results.sort_by_key(|(i, _)| *i);

    on_status("♪ All voices complete — merging...");

    // 3. Merge results
    let merged = merge(&results, &subtasks, task, llm).await?;

    // Safety scan on final output
    let scan = safety.scan_output(&merged);
    if scan.leaks_found {
        return Ok("Ainur response blocked: contains potential secrets.".into());
    }

    Ok(merged)
}

/// Planner: decompose task into N subtasks via LLM.
async fn plan(
    count: usize,
    task: &str,
    llm: &Arc<dyn LlmProvider>,
) -> Result<Vec<String>> {
    let messages = vec![Message {
        role: Role::User,
        content: vec![ContentBlock::text(format!(
            "N={count}\nTask: {task}"
        ))],
    }];

    let response = llm.chat(&messages, &[], PLANNER_PROMPT).await?;

    let text = response
        .content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<String>();

    // Parse JSON array from response
    let text = text.trim();
    // Strip markdown code fences if present
    let text = text
        .strip_prefix("```json")
        .or_else(|| text.strip_prefix("```"))
        .unwrap_or(text);
    let text = text.strip_suffix("```").unwrap_or(text).trim();

    let subtasks: Vec<String> = serde_json::from_str(text).map_err(|e| {
        Error::Tool(format!(
            "planner returned invalid JSON: {e}\nResponse: {text}"
        ))
    })?;

    Ok(subtasks)
}

/// Worker: execute a single subtask with tool access.
async fn run_worker(
    llm: &Arc<dyn LlmProvider>,
    subtask: &str,
    tools: &ToolRegistry,
    tool_defs: &[ToolDef],
    system: &str,
    max_turns: usize,
) -> Result<String> {
    let mut messages = vec![Message {
        role: Role::User,
        content: vec![ContentBlock::text(subtask)],
    }];

    let mut turns = 0;
    loop {
        if turns >= max_turns {
            return Ok("Worker reached tool use limit.".into());
        }
        turns += 1;

        let response = llm.chat(&messages, tool_defs, system).await?;

        let mut tool_uses = Vec::new();
        let mut text_parts = Vec::new();
        let mut assistant_blocks = Vec::new();

        for block in &response.content {
            match block {
                ContentBlock::Text { text } => {
                    text_parts.push(text.clone());
                    assistant_blocks.push(block.clone());
                }
                ContentBlock::ToolUse { id, name, input } => {
                    tool_uses.push((id.clone(), name.clone(), input.clone()));
                    assistant_blocks.push(block.clone());
                }
                _ => {
                    assistant_blocks.push(block.clone());
                }
            }
        }

        messages.push(Message {
            role: Role::Assistant,
            content: assistant_blocks,
        });

        if tool_uses.is_empty() || response.stop_reason != StopReason::ToolUse {
            return Ok(text_parts.join(""));
        }

        // Execute tools
        let mut result_blocks = Vec::new();
        for (id, name, input) in &tool_uses {
            let result = match tools.get(name) {
                Some(tool) => match tool.execute(input.clone()).await {
                    Ok(output) => ContentBlock::tool_result(id, &output),
                    Err(e) => ContentBlock::tool_error(id, e.to_string()),
                },
                None => ContentBlock::tool_error(id, format!("Unknown tool: {name}")),
            };
            result_blocks.push(result);
        }

        messages.push(Message {
            role: Role::User,
            content: result_blocks,
        });
    }
}

/// Max characters per worker result in the merge prompt.
/// Keeps total merge input under ~8K tokens even with 10 workers.
const MAX_WORKER_CHARS: usize = 2000;

/// Merger: combine worker results into a cohesive response.
async fn merge(
    results: &[(usize, String)],
    subtasks: &[String],
    original_task: &str,
    llm: &Arc<dyn LlmProvider>,
) -> Result<String> {
    let mut worker_text = String::new();
    for (i, (_, result)) in results.iter().enumerate() {
        let trimmed = if result.len() > MAX_WORKER_CHARS {
            format!("{}...(truncated)", &result[..MAX_WORKER_CHARS])
        } else {
            result.clone()
        };
        worker_text.push_str(&format!(
            "## Worker {} — {}\n{}\n\n",
            i + 1,
            subtasks.get(i).map(|s| s.as_str()).unwrap_or(""),
            trimmed,
        ));
    }

    let messages = vec![Message {
        role: Role::User,
        content: vec![ContentBlock::text(format!(
            "Original task: {original_task}\n\n\
             Worker results:\n\n{worker_text}\n\
             Combine these into a single cohesive response."
        ))],
    }];

    // Retry once on rate limit (workers may have exhausted the budget)
    let response = match llm.chat(&messages, &[], MERGER_PROMPT).await {
        Ok(r) => r,
        Err(e) if e.to_string().contains("rate limit") => {
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            llm.chat(&messages, &[], MERGER_PROMPT).await?
        }
        Err(e) => return Err(e),
    };

    let text = response
        .content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<String>();

    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ainur_basic() {
        let (count, task) = parse_ainur("/ainur 3 compare cloud providers").unwrap();
        assert_eq!(count, 3);
        assert_eq!(task, "compare cloud providers");
    }

    #[test]
    fn test_parse_ainur_single() {
        let (count, task) = parse_ainur("/ainur 1 do something").unwrap();
        assert_eq!(count, 1);
        assert_eq!(task, "do something");
    }

    #[test]
    fn test_parse_ainur_max() {
        let (count, _) = parse_ainur("/ainur 10 big task").unwrap();
        assert_eq!(count, 10);
    }

    #[test]
    fn test_parse_ainur_too_many() {
        assert!(parse_ainur("/ainur 11 too many").is_none());
    }

    #[test]
    fn test_parse_ainur_zero() {
        assert!(parse_ainur("/ainur 0 nothing").is_none());
    }

    #[test]
    fn test_parse_ainur_no_task() {
        assert!(parse_ainur("/ainur 3 ").is_none());
    }

    #[test]
    fn test_parse_ainur_no_count() {
        assert!(parse_ainur("/ainur abc task").is_none());
    }

    #[test]
    fn test_parse_ainur_wrong_prefix() {
        assert!(parse_ainur("/spawn 3 task").is_none());
    }
}
