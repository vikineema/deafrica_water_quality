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
      buildInputs = [ micromamba-shell.packages.${system}.default pkgs.jq pkgs.moreutils ];

      shellHook = 
        ''
          if [ -z "$MICROMAMBA_EXE" ]; then
            echo "Entering default micromamba shell..."
            exec micromamba-shell --rcfile ${./micromamba_shell_hook.sh}
          fi
        '';
    };
  };
}
