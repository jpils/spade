{
	description = "Rust dev env";

	inputs = {
		nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

		flake-utils.url = "github:numtide/flake-utils";

		fenix = {
			url = "github:nix-community/fenix";
			inputs.nixpkgs.follows = "nixpkgs";
		};

		crane.url = "github:ipetkov/crane";

		my-pkgs.url = "github:jpils/nixconf";
	};

	outputs = { self, nixpkgs, flake-utils, fenix, crane, my-pkgs, ... }:
		flake-utils.lib.eachDefaultSystem (system:
			let
				pkgs = import nixpkgs { inherit system; };

				tc = fenix.packages.${system}.stable;
				toolchain = tc.withComponents [
					"cargo"
					"clippy"
					"rust-src"
					"rustc"
					"rustfmt"
					"rust-analyzer"
				];

				craneLib = (crane.mkLib pkgs).overrideToolchain toolchain;
				src = craneLib.cleanCargoSource ./.;
				commonArgs = {
					inherit src;
					strictDeps = true;
				};

				cargoArtifacts = craneLib.buildDepsOnly commonArgs;
				crate = craneLib.buildPackage (commonArgs // { inherit cargoArtifacts; });

				neovim = my-pkgs.packages.${system}.neovim;
			in
				{
				devShells.default = pkgs.mkShell {
					packages = [
						toolchain
						fenix.packages.${system}.rust-analyzer
						neovim
						pkgs.bacon
					];

					RUST_SRC_PATH = "${tc.rust-src}/lib/rustlib/src/rust/library";
				};


				packages.default = crate;
			});
}
