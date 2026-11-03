use std::path::PathBuf;

use clap::{Parser, Subcommand, Args};

#[derive(Parser, Debug)]
#[command(
    name = "spade", 
    version = "0.1.0", 
    about = "Structural Perovskite Analysis of Domain Wall Environments"
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Commands
}

#[derive(Subcommand, Debug)]
pub(crate) enum Commands {
    Ananlyze(AnalyzeArgs)
}

#[derive(Args, Debug)]
pub(crate) struct AnalyzeArgs {
    #[arg(short, long)]
    pub(crate) config: PathBuf,

    #[arg(short, long, default_value = "result")]
    pub(crate) output: PathBuf
}
