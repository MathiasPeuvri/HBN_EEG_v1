# 1. Train your models (you've done this with CRL pretraining)
  python src/ML_pipeline_test/crl_pretraining.py

  # 2. Train downstream regressors for both challenges
  python regression.py --encoder_type crl --target response_time  # Challenge 1
  python regression.py --encoder_type crl --target externalizing   # Challenge 2

  # 3. Adapt submission.py to load your models
  # (we can do this together)

  # 4. Create submission zip
  python src/ML_pipeline_test/starterkit/generate_submission_zip.py \
      src/ML_pipeline_test/starterkit/submission.py \
      src/ML_pipeline_test/saved_models/crl_encoder_best.pth \
      src/ML_pipeline_test/saved_models/regressor_challenge1_best.pth \
      src/ML_pipeline_test/saved_models/regressor_challenge2_best.pth

  # 5. Test locally

  python src/ML_pipeline_test/starterkit/startkit_localscoring.py \
      --submission-zip submission_10_14_20_*.zip \
      --data-dir /home/mts/HBN_EEG_v1/database \
      --output-dir ./test_submission_output \
      --fast-dev-run