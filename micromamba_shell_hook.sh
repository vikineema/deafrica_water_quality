#!/usr/bin/env bash
# ---------------- FHS micromamba shell ----------------
ENV_NAME="deafrica-water-quality-env"
PROD_YAML="docker/environment_prod.yaml"
DEV_YAML="docker/environment_dev.yaml"

# Keep record of the env files hash files
PROD_ENV_HASH=$(sha256sum "$PROD_YAML" | awk '{print $1}')
DEV_ENV_HASH=$(sha256sum "$DEV_YAML" | awk '{print $1}')

HASH_FILE=".tmp/${ENV_NAME}_hash.json"
mkdir -p "$(dirname "$HASH_FILE")"

# Auto-create or update environment
if ! micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creating micromamba environment '$ENV_NAME'..."
    micromamba create -n "$ENV_NAME" -f "$PROD_YAML" -y
    micromamba install -n "$ENV_NAME" -f "$DEV_YAML" -y
    jq -n \
        --arg PROD_ENV_HASH "$PROD_ENV_HASH" \
        --arg DEV_ENV_HASH "$DEV_ENV_HASH" \
        '{PROD_ENV_HASH: $PROD_ENV_HASH, DEV_ENV_HASH: $DEV_ENV_HASH}' > "$HASH_FILE"
else
    # Update PROD_ENV_HASH if changed
    if [ ! -f "$HASH_FILE" ] || [ "$PROD_ENV_HASH" != "$(jq -r '.PROD_ENV_HASH' "$HASH_FILE")" ]; then
        echo "Updating PROD dependencies in micromamba environment '$ENV_NAME'..."
        micromamba install -n "$ENV_NAME" -f "$PROD_YAML" -y
        jq --arg new "$PROD_ENV_HASH" '.PROD_ENV_HASH = $new' "$HASH_FILE" | sponge "$HASH_FILE"
    fi

    # Update DEV_ENV_HASH if changed
    if [ ! -f "$HASH_FILE" ] || [ "$DEV_ENV_HASH" != "$(jq -r '.DEV_ENV_HASH' "$HASH_FILE")" ]; then
        echo "Updating DEV dependencies in micromamba environment '$ENV_NAME'..."
        micromamba install -n "$ENV_NAME" -f "$DEV_YAML" -y
        jq --arg new "$DEV_ENV_HASH" '.DEV_ENV_HASH = $new' "$HASH_FILE" | sponge "$HASH_FILE"
    fi
fi

# Activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate "$ENV_NAME"

# Start interactive shell
exec bash
