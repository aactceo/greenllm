---
name: green-ai
description: Energy-efficient way of working for any coding or writing task. Use on every task to cut the tokens and model calls needed (and so the energy and carbon) without lowering quality - targeted reads, no repeated work, concise answers, smallest adequate model.
---

# Green AI: do the same work with fewer tokens

Energy used by a language model grows with the size of the model, the number of
tokens it processes (input and output) and the number of calls. Each rule below
lowers one of those. Quality comes first: never skip work that is needed for a
correct result.

## Reading and context (input tokens: usually the largest share)
- Search before you read: use Grep/Glob to find the exact place, then read only the
  relevant line range (`offset`/`limit`) instead of whole files.
- Do not re-read a file you already read in this conversation unless it changed.
- Do not paste large outputs back into the conversation. Filter command output
  (`| head`, `| tail`, `grep`, `--quiet`) to what you need.
- Keep stable instructions (system prompt, long reference docs) at the start and
  unchanged, so the provider's prompt cache can reuse them.

## Calls and tools (number of calls)
- Batch independent tool calls in one turn instead of one per turn.
- Run the smallest relevant test target first, not the whole suite each time.
- Do not start subagents for work you can do directly; each starts with a fresh,
  full context.
- Stop when the task is done. Do not add unrequested features, refactors or
  extra files.

## Output (output tokens: the most expensive per token)
- Answer directly. No restating the question, no long preambles or summaries of
  what you are about to do.
- Show only changed code, not whole files, unless asked.
- Prefer one clear recommendation over a long list of options.

## Model choice
- When the work is simple (renames, formatting, short lookups, classification,
  boilerplate), say once that a smaller model (for example a Haiku-class model)
  is enough, and suggest switching with `/model`.
- Keep large models for hard reasoning, design and difficult debugging.

## Non-urgent bulk work
- For large offline jobs (bulk classification, summarising many documents),
  suggest the provider's Batch API and, where the timing is flexible, running at
  hours when the local grid is cleaner.

## Gateway
If the user runs the GreenLLM gateway (`greenllm serve`), the session's real
token savings are at `http://localhost:8787/` and `/stats`. Quote those numbers
rather than guessing.
