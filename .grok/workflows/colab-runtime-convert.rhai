let meta = #{
    name: "colab-runtime-convert",
    description: "Convert AMLK training runtime from HF Jobs to google-colab-cli via trial-and-error small experiments, preserving Hub/wandb/prediction save contracts",
    when_to_use: "When migrating AMLK cloud training/inference to Colab CLI, validating colab run/exec, or proving smoke training still pushes the same Hub artifacts",
    phases: [
        #{ title: "Inventory", detail: "map HF Jobs save contracts + Colab CLI surface" },
        #{ title: "Explore", detail: "trial-and-error Colab CLI CPU exercises" },
        #{ title: "Design", detail: "Colab training path that preserves prior saves" },
        #{ title: "Implement", detail: "code path: submit-colab + remote runner" },
        #{ title: "CPU-Smokes", detail: "data/bootstrap/file-transfer experiments" },
        #{ title: "Train-Smoke", detail: "tiny GPU training smoke with Hub push" },
        #{ title: "Verify", detail: "artifacts + skill updates + report" },
    ],
};

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------
let inventory_schema = #{
    "type": "object",
    "required": ["hf_save_contract", "colab_surface", "gaps", "risks"],
    "properties": #{
        "hf_save_contract": #{ "type": "string" },
        "colab_surface": #{ "type": "string" },
        "gaps": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 12 },
        "risks": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 10 },
        "must_preserve": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 15 },
    },
};

let explore_schema = #{
    "type": "object",
    "required": ["version", "auth_ok", "lessons", "cpu_smoke_ok", "sessions_cleaned", "commands_proven"],
    "properties": #{
        "version": #{ "type": "string" },
        "auth_ok": #{ "type": "boolean" },
        "auth_mode": #{ "type": "string" },
        "cpu_smoke_ok": #{ "type": "boolean" },
        "sessions_cleaned": #{ "type": "boolean" },
        "commands_proven": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 30 },
        "failures_and_fixes": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 20 },
        "lessons": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 20 },
        "skill_drift_notes": #{ "type": "string" },
    },
};

let design_schema = #{
    "type": "object",
    "required": ["approach", "files_to_touch", "save_mapping", "smoke_plan", "non_goals"],
    "properties": #{
        "approach": #{ "type": "string" },
        "files_to_touch": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 12 },
        "save_mapping": #{ "type": "string" },
        "env_vars": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 25 },
        "smoke_plan": #{ "type": "string" },
        "non_goals": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 10 },
        "colab_session_strategy": #{ "type": "string" },
    },
};

let implement_schema = #{
    "type": "object",
    "required": ["success", "files_changed", "how_to_run", "notes"],
    "properties": #{
        "success": #{ "type": "boolean" },
        "files_changed": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 20 },
        "how_to_run": #{ "type": "string" },
        "notes": #{ "type": "string" },
        "blockers": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 10 },
    },
};

let smoke_schema = #{
    "type": "object",
    "required": ["success", "experiments", "artifacts", "failures_and_fixes", "cleanup_done"],
    "properties": #{
        "success": #{ "type": "boolean" },
        "experiments": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 20 },
        "artifacts": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 20 },
        "failures_and_fixes": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 20 },
        "cleanup_done": #{ "type": "boolean" },
        "hub_repos": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 10 },
        "notes": #{ "type": "string" },
    },
};

let verify_schema = #{
    "type": "object",
    "required": ["contract_ok", "report_path", "summary", "remaining_work"],
    "properties": #{
        "contract_ok": #{ "type": "boolean" },
        "report_path": #{ "type": "string" },
        "summary": #{ "type": "string" },
        "remaining_work": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 15 },
        "docs_updated": #{ "type": "array", "items": #{ "type": "string" }, "maxItems": 10 },
    },
};

// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------
// Guard `args` first: property access on unit throws.
let skip_gpu = false;
let gpu = "T4";
let hf_user = "avreymi";
let smoke_repo_suffix = "colab-smoke";
if args != () {
    if args.skip_gpu != () { skip_gpu = args.skip_gpu; }
    if args.gpu != () { gpu = args.gpu; }
    if args.hf_user != () { hf_user = args.hf_user; }
    if args.smoke_repo_suffix != () { smoke_repo_suffix = args.smoke_repo_suffix; }
}

log("colab-runtime-convert: skip_gpu=" + skip_gpu.to_string()
    + " gpu=" + gpu + " hf_user=" + hf_user
    + " smoke_repo_suffix=" + smoke_repo_suffix);

// ---------------------------------------------------------------------------
// Phase 1 — Inventory (parallel read-only)
// ---------------------------------------------------------------------------
phase("Inventory");

let inv_jobs = [];

let p_hf = "";
p_hf += "AMLK inventory (READ-ONLY). Map the CURRENT HuggingFace Jobs training save contract.\n";
p_hf += "Repo root is the workspace. Use read_file/grep/list_dir — do not edit files.\n\n";
p_hf += "Read and synthesize from:\n";
p_hf += "- training/train.py (submit_hf_job, env, Hub create_repo, upload)\n";
p_hf += "- training/train_hf_job.py (checkpoints, hub_strategy, predictions upload, /data/output)\n";
p_hf += "- training/config.py (MODEL_ID, MODEL_SLUG, METHOD_PRESETS, repo helpers, wandb names)\n";
p_hf += "- training/resume.py (full Trainer checkpoint rules)\n";
p_hf += "- .agents/skills/training/SKILL.md and CLAUDE.md / AGENTS.md Status\n\n";
p_hf += "Report precisely:\n";
p_hf += "1) hf_save_contract: what lands on Hub (adapter root, checkpoint-N contents, predictions-*.jsonl),\n";
p_hf += "   local paths, wandb project/run naming, env vars TRAIN_CONFIG/LORA_CONFIG, soft-fail Hub I/O.\n";
p_hf += "2) must_preserve: bullet list of behaviors that Colab path MUST keep identical.\n";
p_hf += "3) gaps: what HF Jobs gives that Colab lacks (e.g. /data/output bucket mount).\n";
p_hf += "4) risks: silent regressions (double-BOS, chat template, decode penalties, local GPU freeze).\n";
p_hf += "colab_surface can be a short placeholder — the other agent covers CLI details.\n";
p_hf += "Do not invent; quote file paths and flag names from code.\n";

inv_jobs.push(#{
    prompt: p_hf,
    label: "inv-hf-jobs",
    capability_mode: "read-only",
    output_schema: inventory_schema,
});

let p_colab = "";
p_colab += "AMLK inventory (READ-ONLY). Map the google-colab-cli surface for this machine.\n";
p_colab += "Use shell only for non-destructive discovery: colab version, colab help, colab skill,\n";
p_colab += "colab help run/new/exec/upload/download/install/log/sessions, colab --auth=oauth2 whoami,\n";
p_colab += "colab --auth=oauth2 sessions. Do NOT allocate GPU/TPU. Do NOT start long sessions.\n";
p_colab += "Also read .agents/skills/colab-cli/SKILL.md and note version drift vs installed CLI.\n\n";
p_colab += "Fill:\n";
p_colab += "- colab_surface: commands, auth default, file transfer order, colab run vs durable session,\n";
p_colab += "  timeout defaults, agent safety rules.\n";
p_colab += "- gaps/risks: keep-alive 403 history, GenerativeAI import hook, no local GPU, secrets via .env upload.\n";
p_colab += "- hf_save_contract can be brief pointer to training skill.\n";
p_colab += "- must_preserve: Colab operational invariants (always --auth=oauth2 here, always stop sessions).\n";
p_colab += "Learn from actual CLI output; if skill and binary disagree, trust the binary.\n";

inv_jobs.push(#{
    prompt: p_colab,
    label: "inv-colab-cli",
    capability_mode: "execute",
    output_schema: inventory_schema,
});

let inv_results = parallel(inv_jobs);

let hf_contract = "";
let colab_surface = "";
let must_preserve = [];
let inv_gaps = [];
let inv_risks = [];
for r in inv_results {
    if r != () && r.success && r.output != () {
        if r.output.hf_save_contract != () && r.output.hf_save_contract != "" {
            if hf_contract == "" { hf_contract = r.output.hf_save_contract; }
            else { hf_contract += "\n---\n" + r.output.hf_save_contract; }
        }
        if r.output.colab_surface != () && r.output.colab_surface != "" {
            if colab_surface == "" { colab_surface = r.output.colab_surface; }
            else { colab_surface += "\n---\n" + r.output.colab_surface; }
        }
        if r.output.must_preserve != () {
            for m in r.output.must_preserve { must_preserve.push(m); }
        }
        if r.output.gaps != () {
            for g in r.output.gaps { inv_gaps.push(g); }
        }
        if r.output.risks != () {
            for k in r.output.risks { inv_risks.push(k); }
        }
    }
}

if hf_contract == "" {
    await_user("verification",
        "Inventory failed to extract HF Jobs save contract. Resume after fixing auth/tools, or abort.");
}

log("Inventory done: preserve_items=" + must_preserve.len().to_string()
    + " gaps=" + inv_gaps.len().to_string());

// ---------------------------------------------------------------------------
// Phase 2 — Explore Colab CLI (trial and error, CPU only)
// ---------------------------------------------------------------------------
phase("Explore");

let p_explore = "";
p_explore += "You are converting AMLK to google-colab-cli. EXPLORE the package by trial and error on CPU only.\n";
p_explore += "Goal: prove agent-safe CLI patterns for this repo. Never allocate GPU/TPU in this phase.\n";
p_explore += "Always pass --auth=oauth2 before the subcommand unless help proves oauth2 is default AND whoami works bare.\n";
p_explore += "Use an isolated config: --config /tmp/amlk-colab-explore.json for all session commands.\n\n";
p_explore += "Mandatory exercises (run real commands; record failures and fixes):\n";
p_explore += "1) colab version; colab update (check only); colab help; colab skill | head.\n";
p_explore += "2) colab --auth=oauth2 whoami; colab --auth=oauth2 sessions.\n";
p_explore += "3) Preflight: colab run on a missing script — must exit before allocating.\n";
p_explore += "4) Write /tmp/amlk_colab_hello.py that prints platform, cwd, python version, and writes /content/hello_out.txt.\n";
p_explore += "5) colab --auth=oauth2 --config /tmp/amlk-colab-explore.json run --timeout 120 /tmp/amlk_colab_hello.py\n";
p_explore += "   (prefer run so VM self-stops; if run cannot download outputs, use durable session next).\n";
p_explore += "6) Durable CPU session: new -s amlk-cpu-explore; status; exec print; upload a tiny file;\n";
p_explore += "   ls; download; install packaging (or a tiny pkg); restart-kernel; log -n 20; stop.\n";
p_explore += "7) After stop, sessions must show clean (or document orphans + how you cleared them).\n";
p_explore += "8) Compare installed CLI vs .agents/skills/colab-cli/SKILL.md; list skill_drift_notes.\n\n";
p_explore += "Hard rules:\n";
p_explore += "- NEVER unpiped repl/console (hang).\n";
p_explore += "- NEVER leave a session running when done — stop and verify.\n";
p_explore += "- Do not train models or load 7B weights.\n";
p_explore += "- If a command fails, try one alternate pattern and document both.\n\n";
p_explore += "Return structured fields: version, auth_ok, auth_mode, cpu_smoke_ok, sessions_cleaned,\n";
p_explore += "commands_proven, failures_and_fixes, lessons, skill_drift_notes.\n";

let explore_r = agent(p_explore, #{
    label: "explore-colab-cli",
    capability_mode: "all",
    output_schema: explore_schema,
});

if explore_r == () || !explore_r.success || explore_r.output == () {
    await_user("infra",
        "Colab CLI explore agent failed. Check oauth2 auth / colab install, then resume.");
}

if explore_r.output.auth_ok != true {
    await_user("infra",
        "Colab auth failed (auth_ok=false). Fix OAuth/ADC, then resume this workflow.");
}

if explore_r.output.sessions_cleaned != true {
    log("WARNING: explore phase may have left sessions — Train-Smoke will re-check cleanup");
}

let explore_lessons = "";
if explore_r.output.lessons != () {
    for L in explore_r.output.lessons {
        explore_lessons += "- " + L + "\n";
    }
}
let explore_fails = "";
if explore_r.output.failures_and_fixes != () {
    for f in explore_r.output.failures_and_fixes {
        explore_fails += "- " + f + "\n";
    }
}

log("Explore: version=" + explore_r.output.version.to_string()
    + " cpu_smoke_ok=" + explore_r.output.cpu_smoke_ok.to_string());

// ---------------------------------------------------------------------------
// Phase 3 — Design
// ---------------------------------------------------------------------------
phase("Design");

let p_design = "";
p_design += "Design the AMLK Colab training runtime path. READ existing code; do not implement yet.\n\n";
p_design += "HF Jobs save contract (from inventory):\n";
p_design += json_encode(hf_contract) + "\n\n";
p_design += "Must preserve (array):\n";
p_design += json_encode(must_preserve) + "\n\n";
p_design += "Colab surface / explore lessons:\n";
p_design += json_encode(colab_surface) + "\n";
p_design += "Explore lessons:\n" + explore_lessons + "\n";
p_design += "Explore failures/fixes:\n" + explore_fails + "\n\n";
p_design += "Constraints:\n";
p_design += "- Never train on local 8GB GPU (freezes). All model work on Colab GPU or keep HF Jobs.\n";
p_design += "- Prefer reusing training/train_hf_job.py logic OR training.train local path + Hub push —\n";
p_design += "  do NOT fork hyperparams; METHOD_PRESETS / TRAIN_CONFIG / LORA_CONFIG stay source of truth.\n";
p_design += "- Saves must land the SAME way as before: private Hub model repo with adapter + full\n";
p_design += "  checkpoint-* (optimizer+trainer_state when using hub_strategy=all_checkpoints),\n";
p_design += "  predictions-finetuned.jsonl / predictions-base.jsonl, wandb project amlk-dictalm2-instruct\n";
p_design += "  with informative run names, create_repo before train so mid-run pushes work.\n";
p_design += "- Replace /data/output (HF Jobs bucket) with a Colab path (e.g. /content/amlk-output) that\n";
p_design += "  still supports resume_from_checkpoint within a session; cross-job resume remains Hub-based.\n";
p_design += "- Secrets: HF_TOKEN, WANDB_API_KEY from local .env — upload or inject carefully; never log secrets.\n";
p_design += "- Small experiments first: smoke-test (10 steps), max-train tiny, skip-base-arm optional,\n";
p_design += "  dedicated smoke output-repo avreymi/amlk-dictalm2-instruct-" + smoke_repo_suffix + ".\n";
p_design += "- CLI: --auth=oauth2; prefer colab run for unattended; durable session only if multi-step setup.\n";
p_design += "- Gemini import hook gotcha if any judge code runs on Colab.\n";
p_design += "- Keep --submit-hf working; Colab is additive (e.g. --submit-colab) not a break of HF path.\n\n";
p_design += "hf_user=" + hf_user + " gpu_default=" + gpu + " skip_gpu=" + skip_gpu.to_string() + "\n\n";
p_design += "Return: approach, files_to_touch, save_mapping (old path → new path), env_vars, smoke_plan,\n";
p_design += "non_goals, colab_session_strategy.\n";

let design_r = agent(p_design, #{
    label: "design-colab-path",
    capability_mode: "read-only",
    output_schema: design_schema,
});

if design_r == () || !design_r.success || design_r.output == () {
    await_user("verification", "Design agent failed. Resume to retry design, or abort.");
}

log("Design approach ready; files_to_touch count will drive implement");

// ---------------------------------------------------------------------------
// Phase 4 — Implement
// ---------------------------------------------------------------------------
phase("Implement");

let p_impl = "";
p_impl += "IMPLEMENT the AMLK Colab training submission path. Follow the approved design and project rules.\n\n";
p_impl += "Design JSON:\n";
p_impl += json_encode(design_r.output) + "\n\n";
p_impl += "Must preserve saves:\n";
p_impl += json_encode(must_preserve) + "\n\n";
p_impl += "Explore lessons (obey them):\n" + explore_lessons + "\n\n";
p_impl += "Requirements:\n";
p_impl += "1) Add a first-class way to launch training on Colab from the repo root, e.g.\n";
p_impl += "   python -m training.train --submit-colab --hf-user " + hf_user + " --smoke-test ...\n";
p_impl += "   OR a dedicated module training/submit_colab.py called by train.py — pick the design's approach.\n";
p_impl += "2) Keep --submit-hf intact. Shared helpers for env/config serialization are fine.\n";
p_impl += "3) Remote code must still: chat-wrap prompts, completion_only_loss, Hebrew bad_words_ids,\n";
p_impl += "   decode defaults repetition_penalty=1.0 / no_repeat_ngram_size=0, hub push of adapter +\n";
p_impl += "   full checkpoints + predictions JSONL, wandb naming via training.config helpers.\n";
p_impl += "4) Colab output dir: not /data/output — use /content/... and document it. Hub remains\n";
p_impl += "   the cross-session durability layer (same as after HF Jobs SIGTERM).\n";
p_impl += "5) File headers mandatory (project architecture + code flow + execution env).\n";
p_impl += "6) Read .agents/skills/coding-principles/SKILL.md and follow it.\n";
p_impl += "7) Prefer minimal diff. Reuse train_hf_job.py if possible (env OUTPUT_DIR override) rather\n";
p_impl += "   than duplicating 500 lines.\n";
p_impl += "8) Support smoke-test, skip-data-upload, output-repo, method, dataset-repo overrides.\n";
p_impl += "9) After coding: run local unit tests that don't need GPU (pytest subset if exists) and a\n";
p_impl += "   dry CLI --help / argparse path for --submit-colab. Do NOT load 7B locally.\n";
p_impl += "10) If blocked (missing deps), set success=false and list blockers — do not fake Hub pushes.\n\n";
p_impl += "Return success, files_changed, how_to_run (exact commands), notes, blockers.\n";

let impl_r = agent(p_impl, #{
    label: "implement-colab-train",
    capability_mode: "all",
    output_schema: implement_schema,
});

if impl_r == () || !impl_r.success || impl_r.output == () {
    await_user("verification", "Implement agent failed hard. Resume to retry, or abort.");
}

if impl_r.output.success != true {
    log("Implement reported success=false; blockers will gate later phases");
    await_user("verification",
        "Implementation incomplete (success=false). Inspect blockers, fix or resume to continue with partial path.");
}

log("Implement how_to_run captured");

// ---------------------------------------------------------------------------
// Phase 5 — CPU smokes (everything that is not full GPU train)
// ---------------------------------------------------------------------------
phase("CPU-Smokes");

let p_cpu = "";
p_cpu += "Run SMALL CPU experiments for the new Colab runtime path. Trial and error; fix as you go.\n\n";
p_cpu += "how_to_run from implementer:\n";
p_cpu += json_encode(impl_r.output.how_to_run) + "\n\n";
p_cpu += "files_changed:\n";
p_cpu += json_encode(impl_r.output.files_changed) + "\n\n";
p_cpu += "Experiments (CPU only — no --gpu):\n";
p_cpu += "A) Local: source .env if present; python -m training.train --help must show Colab submit flag.\n";
p_cpu += "B) Local: data path smoke if data exists — python -m data.download may be heavy; prefer checking\n";
p_cpu += "   that processed splits or Hub dataset config resolve without loading models.\n";
p_cpu += "C) Colab CPU: bootstrap experiment — upload/tarball or git clone strategy from design,\n";
p_cpu += "   inject secrets from .env WITHOUT printing them, install minimal deps, import training.config,\n";
p_cpu += "   print MODEL_ID/MODEL_SLUG/MAX_LENGTH. Use --config /tmp/amlk-colab-cpu.json and session name\n";
p_cpu += "   amlk-cpu-smoke-convert. Prefer colab run for one-shots; durable session if multi-step.\n";
p_cpu += "D) File transfer: push a tiny JSON marker, download it back, assert bytes.\n";
p_cpu += "E) If submit-colab has a --dry-run or preflight, run it.\n";
p_cpu += "F) ALWAYS stop sessions; colab --auth=oauth2 sessions must be clean at end.\n";
p_cpu += "G) Record every failure and the fix that worked.\n\n";
p_cpu += "Never load dictalm2 weights on local machine. Never allocate GPU here.\n";
p_cpu += "Return success, experiments list, artifacts paths, failures_and_fixes, cleanup_done, notes.\n";

let cpu_r = agent(p_cpu, #{
    label: "cpu-smokes",
    capability_mode: "all",
    output_schema: smoke_schema,
});

if cpu_r == () || !cpu_r.success || cpu_r.output == () {
    await_user("verification", "CPU smoke agent failed. Resume after checking Colab CPU quota/auth.");
}

if cpu_r.output.cleanup_done != true {
    log("CPU smoke left sessions — forcing cleanup note into train phase");
}

if cpu_r.output.success != true {
    await_user("verification",
        "CPU smokes did not fully succeed. Review failures; resume to re-run or proceed carefully.");
}

log("CPU smokes success=" + cpu_r.output.success.to_string());

// ---------------------------------------------------------------------------
// Phase 6 — Train smoke (GPU, optional)
// ---------------------------------------------------------------------------
phase("Train-Smoke");

let train_r = ();
if skip_gpu {
    log("skip_gpu=true — skipping GPU train smoke");
} else {
    let p_train = "";
    p_train += "Run a TINY Colab GPU training smoke for AMLK. Preserve save contracts exactly.\n\n";
    p_train += "how_to_run:\n";
    p_train += json_encode(impl_r.output.how_to_run) + "\n\n";
    p_train += "Design smoke_plan:\n";
    p_train += json_encode(design_r.output.smoke_plan) + "\n\n";
    p_train += "CPU smoke notes/failures to avoid repeating:\n";
    p_train += json_encode(cpu_r.output) + "\n\n";
    p_train += "Hard requirements:\n";
    p_train += "- GPU: --gpu " + gpu + " (if allocation fails, fall back once to T4, then document and stop).\n";
    p_train += "- Use smoke-test / max-train tiny / few steps — NOT a full epoch.\n";
    p_train += "- output-repo MUST be a dedicated smoke repo, e.g. "
        + hf_user + "/amlk-dictalm2-instruct-" + smoke_repo_suffix + " (private).\n";
    p_train += "- Prefer --skip-data-upload if avreymi/amlk-training-data (or design's dataset) already on Hub.\n";
    p_train += "- Method: qlora if T4 memory tight, else design default (lora). Do not OOM-loop endlessly.\n";
    p_train += "- Must set HF_TOKEN and WANDB_API_KEY on the VM from local .env (never print full tokens).\n";
    p_train += "- After train: verify Hub has adapter files and/or checkpoint-* and preferably\n";
    p_train += "  predictions-finetuned.jsonl (smoke may have tiny test subset).\n";
    p_train += "- Use hf CLI or huggingface_hub to list repo files as proof.\n";
    p_train += "- Download predictions or a checkpoint listing locally under outputs/results/colab-smoke/ if possible.\n";
    p_train += "- ALWAYS stop Colab sessions; verify sessions empty (or only unrelated orphans documented).\n";
    p_train += "- Auth: --auth=oauth2; isolate --config /tmp/amlk-colab-train.json.\n";
    p_train += "- If GPU quota is zero: set success=false, explain, do NOT burn endless retries.\n";
    p_train += "- Learn from trial and error: one config change at a time; log failures_and_fixes.\n\n";
    p_train += "Return success, experiments, artifacts, hub_repos, failures_and_fixes, cleanup_done, notes.\n";

    train_r = agent(p_train, #{
        label: "train-smoke-gpu",
        capability_mode: "all",
        output_schema: smoke_schema,
    });

    if train_r == () || !train_r.success || train_r.output == () {
        await_user("infra", "GPU train smoke agent failed. Check GPU quota/credits; resume or set skip_gpu.");
    }

    if train_r.output.cleanup_done != true {
        // Best-effort cleanup agent
        let p_clean = "";
        p_clean += "Cleanup only. List colab --auth=oauth2 sessions and stop any amlk-* or orphan sessions\n";
        p_clean += "from this conversion work (names like amlk-cpu-*, amlk-train-*, run-*).\n";
        p_clean += "Do not stop sessions that clearly belong to another user task if named differently and busy\n";
        p_clean += "with non-AMLK work — if unsure, list them in notes and stop only amlk-* convert sessions.\n";
        p_clean += "Verify final sessions list. Return success=true if cleaned or nothing to clean.\n";
        p_clean += "experiments/artifacts can be empty lists; cleanup_done must reflect reality.\n";
        let clean_r = agent(p_clean, #{
            label: "cleanup-sessions",
            capability_mode: "execute",
            output_schema: smoke_schema,
        });
        if clean_r != () && clean_r.success {
            log("Cleanup agent ran");
        }
    }

    if train_r.output.success != true {
        await_user("verification",
            "GPU train smoke did not succeed. You may resume after fixing quota/code, or continue to Verify with failure recorded.");
    }

    log("Train smoke success=" + train_r.output.success.to_string());
}

// ---------------------------------------------------------------------------
// Phase 7 — Verify + document
// ---------------------------------------------------------------------------
phase("Verify");

let p_verify = "";
p_verify += "Verify AMLK Colab runtime conversion and write the final report.\n\n";
p_verify += "Inventory must_preserve:\n";
p_verify += json_encode(must_preserve) + "\n\n";
p_verify += "Implement result:\n";
p_verify += json_encode(impl_r.output) + "\n\n";
p_verify += "CPU smokes:\n";
p_verify += json_encode(cpu_r.output) + "\n\n";
p_verify += "Train smoke (may be unit if skipped):\n";
p_verify += json_encode(train_r) + "\n\n";
p_verify += "Explore skill_drift / lessons:\n" + explore_lessons + "\n" + explore_fails + "\n\n";
p_verify += "Tasks:\n";
p_verify += "1) Diff-check that --submit-hf still exists and behavior unchanged at argparse level.\n";
p_verify += "2) Confirm save contract: Hub push paths, prediction filenames, wandb naming, OUTPUT_DIR on Colab.\n";
p_verify += "3) Update .agents/skills/colab-cli/SKILL.md with version drift lessons (installed version,\n";
p_verify += "   auth default reality, AMLK --submit-colab how-to, always-stop, smoke repo naming).\n";
p_verify += "4) Update .agents/skills/training/SKILL.md with Colab path alongside HF Jobs (short section).\n";
p_verify += "5) Update CLAUDE.md / AGENTS.md Status with a dated note about Colab runtime path + smoke results\n";
p_verify += "   if those files are the project status source of truth (they are — CLAUDE.md symlink to AGENTS.md).\n";
p_verify += "6) Write outputs/results/colab-runtime-convert-report.md summarizing:\n";
p_verify += "   approach, commands, experiments, Hub artifacts, failures/fixes, remaining work.\n";
p_verify += "7) Run a final colab --auth=oauth2 sessions check; stop stragglers named amlk-*.\n";
p_verify += "8) contract_ok=true only if implement succeeded AND (skip_gpu OR train smoke Hub proof exists)\n";
p_verify += "   AND CPU smokes succeeded AND sessions cleaned.\n";
p_verify += "If skip_gpu was used, contract_ok may be true for code+CPU only — state that GPU unproven.\n\n";
p_verify += "skip_gpu=" + skip_gpu.to_string() + "\n";
p_verify += "Return contract_ok, report_path, summary, remaining_work, docs_updated.\n";

let verify_r = agent(p_verify, #{
    label: "verify-and-docs",
    capability_mode: "all",
    output_schema: verify_schema,
});

if verify_r == () || !verify_r.success || verify_r.output == () {
    complete(#{
        ok: false,
        summary: "Verify agent failed; code may still have been implemented — check git status",
        implement: impl_r.output,
        cpu: cpu_r.output,
        train: train_r,
    });
}

let report_path = verify_r.output.report_path;
if report_path == () || report_path == "" {
    report_path = "outputs/results/colab-runtime-convert-report.md";
}

// Persist a short scratch copy of the summary for the workflow UI
let scratch = "";
scratch += "# Colab runtime convert — workflow result\n\n";
scratch += verify_r.output.summary + "\n\n";
scratch += "report: " + report_path + "\n";
scratch += "contract_ok: " + verify_r.output.contract_ok.to_string() + "\n";
let sp = write_scratch_file("colab-runtime-convert-summary.md", scratch);

complete(#{
    ok: verify_r.output.contract_ok,
    summary: verify_r.output.summary,
    report_path: report_path,
    scratch: sp,
    remaining_work: verify_r.output.remaining_work,
    docs_updated: verify_r.output.docs_updated,
    how_to_run: impl_r.output.how_to_run,
    skip_gpu: skip_gpu,
    gpu: gpu,
});
