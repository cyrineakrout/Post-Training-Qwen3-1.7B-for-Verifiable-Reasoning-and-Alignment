#!/bin/bash
# CS-552 — submit an interactive RCP job with Jupyter Lab.
#
# ============================================================
#  STEP 1  — set your Gaspar username and team group ID below
#            (both REQUIRED)
# ============================================================
#
# Usage:
#   ./submit.sh                # default: 1 GPU, suffix "lab"
#   ./submit.sh train          # custom job suffix
#
# Once the pod is Running, connect in one of these ways:
#
# 1. Jupyter Lab:
#   runai port-forward <job-name> --port 8888:8888
#   then open http://localhost:8888 (token: cs552).
#   This is a command you run after submission; no submit-time service
#   type is needed.
#
# 2. Shell:
#   runai bash <job-name>
#   Useful for CLI work, package checks, nvidia-smi, or debugging without
#   opening Jupyter.
#
# 3. VS Code:
#   Use the Kubernetes extension to attach VS Code to the running pod
#   named <job-name>-0-0. Open your repo folder inside the pod and use
#   Terminal -> New Terminal for a shell in the same environment.
#
# NOTE: GPU count is fixed at 1 (40GB A100). The course quota is one GPU
# per group allocation; asking for more will leave the job stuck Pending.

set -euo pipefail

# ============== EDIT THESE LINES ==============
GASPAR="atarun"              # <-- YOUR GASPAR EPFL username.
GROUP="g75"                  # <-- YOUR TEAM, e.g. g07.
# ==============================================

# Refuse to run with placeholders.
if [[ "${GASPAR}" == "gaspar" || -z "${GASPAR}" ]]; then
    echo "ERROR: edit submit.sh and set GASPAR to your EPFL GASPAR username." >&2
    exit 1
fi

if [[ "${GROUP}" == "gXX" || -z "${GROUP}" ]]; then
    echo "ERROR: edit submit.sh and set GROUP to your team number (e.g. g07)." >&2
    exit 1
fi

GPUS=1   # course cap: 1 GPU
NODE="${NODE:-a100-40g}"  # DO NOT CHANGE: course quota is only available on this node pool
SUFFIX="${1:-lab}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

# Default image with CUDA, PyTorch, and common libraries.
# Override if you use a custom image.
IMAGE="ayushkumartarun/course-cs-552-standard:v1" 

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

# Override these environment variables in your shell if you want to use
# Hugging Face or Weights & Biases with authentication.
HF_TOKEN="${HF_TOKEN:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

# This script does not mount the personal home PVC, so it does not need
# a hard-coded UID/GID. Use /scratch for course work and deliverables.

echo ">>> Submitting ${JOB_NAME}  (1 GPU)"

runai submit \
  --name "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu "${GPUS}" \
  --large-shm \
  --interactive \
  --node-pools "${NODE}" \
  --environment HF_HOME=/scratch/hf_cache \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment HF_TOKEN="${HF_TOKEN:-}" \
  --environment WANDB_API_KEY="${WANDB_API_KEY:-}" \
  --environment WANDB_DIR=/scratch/wandb \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "\
    mkdir -p /scratch/hf_cache /scratch/wandb && \
    cd /scratch && \
    jupyter lab \
      --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
      --ServerApp.root_dir=/scratch \
      --ServerApp.token=\${JUPYTER_TOKEN:-cs552}"

cat <<EOF

>>> Job submitted: ${JOB_NAME}

Watch it start:    runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:       runai logs -f ${JOB_NAME} -p ${PROJECT}
When Running:      runai port-forward ${JOB_NAME} --port 8888:8888 -p ${PROJECT}
Then open:         http://localhost:8888  (token: cs552)
Shell in pod:      runai bash ${JOB_NAME} -p ${PROJECT}
Stop the job:      runai delete job ${JOB_NAME} -p ${PROJECT}

Other connection options:
  Jupyter Lab:
    Wait until the job is Running, use the port-forward command above,
    then open http://localhost:8888.
    This is best for notebooks, plots, and milestone work.

  Shell:
    runai bash ${JOB_NAME} -p ${PROJECT}
    This is useful for nvidia-smi, checking files, installing a temporary
    package, or running scripts without opening Jupyter.

  VS Code:
    In VS Code, install the Microsoft Kubernetes and Remote Development
    extensions. In the Kubernetes sidebar, find pod ${JOB_NAME}-0-0 and
    choose "Attach Visual Studio Code". Then open your repo folder inside
    the pod and use Terminal -> New Terminal for a shell.

WHEN YOU'RE DONE: \`runai delete job ${JOB_NAME} -p ${PROJECT}\`. Idle sessions
take a GPU away from the rest of the course.

Run:AI note:
  If you see a transient [PodGrouperWarning] that says the PodGroup
  "object has been modified", wait and run the describe command again.
  This is a scheduler reconciliation warning, not a submit.sh syntax
  error. The job is only stuck if it remains Pending after a few minutes.

Storage inside the pod:
  /scratch             team scratch — your group's primary workspace, RW
  /shared-ro/datasets  course datasets (read-only)
  /shared-ro/models    course models   (read-only)
  /shared-rw           shared with ALL students — be careful what you write

Deliverable notebooks:
  Keep individual_notebooks/*.ipynb in your git repo and commit them.
  Use /scratch for caches, checkpoints, and large generated files, not
  as the only place where milestone notebooks exist.
EOF
