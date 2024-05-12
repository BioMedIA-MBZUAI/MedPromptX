**MedPromptX-VQA** is built on the codebase of HAIM, which requires MIMIC-CXR and MIMIC-IV databases.

*Note:* You MUST have PhysioNet credentials to be able to use/download the datasets and codebases.

# Links
*HAIM*
https://www.physionet.org/content/haim-multimodal/1.0.1/

*MIMIC-CXR*
https://www.physionet.org/content/mimic-cxr-jpg/2.0.0/

*MIMIC-IV*
https://www.physionet.org/content/mimiciv/1.0/

# Steps
- Fork [HAIM](https://www.physionet.org/content/haim-multimodal/1.0.1/)
- Add the contents of our ```data``` folder to the forked repo.
- Run ```1_filter.ipynb``` to generate ```patients_cxr_diagnoses.csv```
- Run ```2_create_VQA_examples.ipynb``` to merge the data required for generating prompts.
- Run ```3_prompts.py``` to obtain the final prompts required to run MedPromptX.
- The final dataset folder should be named ```prompts_6_shot```, where number 6 refers to the number of candidates.
- Fork [Grounding DINO (GDINO)](https://github.com/IDEA-Research/GroundingDINO)
- Add ```4_mimic_gdino.py``` to the forked repo of *GDINO* under ```demo``` folder. 
- Apply *GDINO* to the images in folder ```prompts_6_shot``` to obtain grounded images. To do that, run the script below:

```
python demo/4_mimic_gdino.py \
-c groundingdino/config/GroundingDINO_SwinT_OGC.py \
-p weights/groundingdino_swint_ogc.pth \
-o "prompts_images_grounded"
```