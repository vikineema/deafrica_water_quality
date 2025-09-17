{
  description = "Water quality python environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    micromamba-shell.url = "github:vikineema/micromamba-shell";
  };

  outputs = { self, nixpkgs, micromamba-shell, ... }@inputs:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; };
  in
  {
    devShells.${system}.default = pkgs.mkShell {
      buildInputs = [ micromamba-shell.packages.${system}.default ];

      shellHook = ''
        # Only exec micromamba-shell if not already inside it
        if [ -z "$MICROMAMBA_EXE" ]; then
          echo "Entering default micromamba shell..."
          exec micromamba-shell --rcfile <(
            cat <<'EOF'
# ---------------- FHS micromamba shell ----------------
ENV_NAME="deafrica-water-quality-env"
PROD_YAML="docker/environment_prod.yaml"
DEV_YAML="docker/environment_dev.yaml"

# Auto-create or update environment
if ! micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creating micromamba environment '$ENV_NAME'..."
    micromamba create -n $ENV_NAME -f $PROD_YAML -y
    [ -f $DEV_YAML ] && micromamba install -n $ENV_NAME -f $DEV_YAML -y
else
    echo "Updating micromamba environment '$ENV_NAME'..."
    micromamba install -n $ENV_NAME -f $PROD_YAML -y
    [ -f $DEV_YAML ] && micromamba install -n $ENV_NAME -f $DEV_YAML -y
fi

# Activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate $ENV_NAME
mic 
# Start interactive shell
exec bash
EOF
          )
        fi
      '';
    };
  };
}
