python local_test_from_json_for_LDCC.py \
--gpu-id 6 \
--image-root /mnt/disk1/liushijie/Process/Data/Public/LDCC/TrainingSet/Sample/aug \
--image-txt-json /mnt/disk1/liushijie/Process/Data/Public/LDCC/TrainingSet/Sample/json_sample/LDCC_train_sampled_seed-42.json \
--output-file /mnt/disk1/liushijie/Process/Data/Public/LDCC/TrainingSet/Sample/generation_test/seed42/CytoGPT_seed42.json

python local_test_from_json_for_LDCC.py \
--gpu-id 6 \
--image-root /mnt/disk1/liushijie/Process/Data/Public/LDCC/TrainingSet/Sample/aug \
--image-txt-json /mnt/disk1/liushijie/Process/Data/Public/LDCC/TrainingSet/Sample/json_sample/LDCC_train_sampled_seed-52.json \
--output-file /mnt/disk1/liushijie/Process/Data/Public/LDCC/TrainingSet/Sample/generation_test/seed42/CytoGPT_seed52.json
