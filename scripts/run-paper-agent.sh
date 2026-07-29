#!/usr/bin/env bash
# Launch Cursor Agent CLI for AMLK paper writing with an optional alternate API key
# (other Cursor account) without changing the IDE login. Context lives in the repo:
# paper/WRITING_HANDOFF.md + paper/main.tex.
#
# Usage:
#   export CURSOR_API_KEY_ALT="cursor_..."   # key from other account (Dashboard → Integrations)
#   ./scripts/run-paper-agent.sh
#
# Optional:
#   PAPER_AGENT_MODEL=claude-sonnet-5-thinking-max ./scripts/run-paper-agent.sh
#   agent --api-key "$CURSOR_API_KEY_ALT" models   # list models for your key
#
# --- Quick start (recommended) ---
# After running this script, your first message can just be the default prompt below
# (already sent automatically), or you can override it inline, e.g.:
#
#   Read paper/WRITING_HANDOFF.md for context, then open paper/main.tex.
#   We are revising the abstract first (lines 43-60). Do not change other sections yet.
#   My remarks: [paste your feedback]
#
# --- Viewing the compiled PDF ---
# The agent cannot reliably pop open a PDF viewer from the CLI. Either open it yourself
# before starting:
#   open -a "Google Chrome" paper/main.pdf
# ...or, since this repo's paper/ has no bib.bib / figures/ committed (main.pdf is stale
# once .tex changes), ask the agent to recompile a throwaway preview with tectonic and
# open that instead — it knows how to work around the missing bib/figures for a preview.
#
# --- Useful @ paths to reference in your first message ---
#   @paper/WRITING_HANDOFF.md
#   @paper/main.tex
#   @docs/obsidian/Experiment Results.md
#
# --- What to avoid ---
#   - Don't say only "open the paper" — say paper/main.tex (source) or paper/main.pdf
#     (compiled), so the agent knows which one you mean.
#   - Say which section ("abstract only") so it doesn't rewrite the whole draft.
#
# --- Template to paste every time ---
#   Context: paper/WRITING_HANDOFF.md
#   File to edit: paper/main.tex
#   Section: abstract (lines 43-60)
#   Task: [your remarks]
#   Rules: edit only the abstract; keep LaTeX valid; don't touch results numbers unless I ask
#
# --- Shortcut ---
# If you paste your remarks into paper/WRITING_HANDOFF.md under "Amit's remarks" first,
# you can shorten every future first message to:
#   Read paper/WRITING_HANDOFF.md and revise the abstract in paper/main.tex accordingly.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

API_KEY="${CURSOR_API_KEY_ALT:-${CURSOR_API_KEY:-}}"
MODEL="${PAPER_AGENT_MODEL:-claude-sonnet-5-thinking-high}"

if [[ -z "$API_KEY" ]]; then
  echo "No API key set. Use one of:"
  echo "  export CURSOR_API_KEY_ALT='cursor_...'   # recommended (other account)"
  echo "  export CURSOR_API_KEY='cursor_...'"
  echo "Get a key: https://cursor.com/dashboard/integrations"
  exit 1
fi

echo "Workspace: $ROOT"
echo "Model:     $MODEL"
echo "Handoff:   paper/WRITING_HANDOFF.md"
echo ""

exec agent \
  --api-key "$API_KEY" \
  --workspace "$ROOT" \
  --trust \
  --model "$MODEL" \
  "Continue AMLK paper writing. Read paper/WRITING_HANDOFF.md first, then paper/main.tex. We are revising section-by-section; start with the abstract unless my remarks are in the handoff file."
