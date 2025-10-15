import zipfile
import datetime
from pathlib import Path
from argparse import ArgumentParser


# usage : 
# python scripts/generate_submission_zip.py src/ML_pipeline_test/starterkit/submission.py  src/ML_pipeline_test/saved_models/crl_encoder_best.pth src/ML_pipeline_test/saved_models/regressor_response_time_crl_frozen_best.pth


def main():
    parser = ArgumentParser()
    parser.add_argument('files', type=Path, nargs='+')
    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime('%m_%d_%H_%M')
    output_fname = f"submission_{timestamp}.zip"

    with zipfile.ZipFile(output_fname, 'w') as myzip:
        for f in args.files:
            myzip.write(f, arcname=f.name)

if __name__ == "__main__":
    main()
