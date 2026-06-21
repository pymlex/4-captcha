import argparse

import _path

from hub.publish import publish_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish outputs to GitHub and artefacts to Hugging Face",
    )
    parser.add_argument(
        "--skip-github",
        action="store_true",
        help="Skip git commit and push of metrics, predictions, plots",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip Hugging Face dataset upload",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip Hugging Face model repo upload",
    )
    parser.add_argument(
        "--rebuild-archive",
        action="store_true",
        help="Rebuild data.tar.gz even if fingerprint matches",
    )
    args = parser.parse_args()
    publish_all(
        github=not args.skip_github,
        hf_dataset=not args.skip_dataset,
        hf_models=not args.skip_models,
        force_archive=args.rebuild_archive,
    )


if __name__ == "__main__":
    main()
