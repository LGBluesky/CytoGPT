
python local_test_from_json_for_final_finetune_model_gen_again.py \
--gpu-id 3 \
--num_beams 1 \
--temperature 0.0 \
--image-root /mnt/disk1/liushijie/Process/Data/generation_evaluation \
--image-txt-json /mnt/disk1/liushijie/Process/Data/generation_evaluation/evaluation_result/CytoGPT/test_extra_seed52-82/ori_data/matched_seed82.json \
--output-file /mnt/disk1/liushijie/Process/Data/generation_evaluation/evaluation_result/CytoGPT/test_extra_seed52-82/seed82/CytoGPT-seed82_t_0_nb_1.json



python local_test_from_json_for_final_finetune_model_gen_again.py \
--gpu-id 3 \
--num_beams 1 \
--temperature 0.5 \
--image-root /mnt/disk1/liushijie/Process/Data/generation_evaluation \
--image-txt-json /mnt/disk1/liushijie/Process/Data/generation_evaluation/evaluation_result/CytoGPT/test_extra_seed52-82/ori_data/matched_seed82.json \
--output-file /mnt/disk1/liushijie/Process/Data/generation_evaluation/evaluation_result/CytoGPT/test_extra_seed52-82/seed82/CytoGPT-seed82_t_0.5_nb_1.json



python local_test_from_json_for_final_finetune_model_gen_again.py \
--gpu-id 3 \
--num_beams 3 \
--temperature 0.5 \
--image-root /mnt/disk1/liushijie/Process/Data/generation_evaluation \
--image-txt-json /mnt/disk1/liushijie/Process/Data/generation_evaluation/evaluation_result/CytoGPT/test_extra_seed52-82/ori_data/matched_seed82.json \
--output-file /mnt/disk1/liushijie/Process/Data/generation_evaluation/evaluation_result/CytoGPT/test_extra_seed52-82/seed82/CytoGPT-seed82_t_0.5_nb_3.json

