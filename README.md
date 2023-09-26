# Document-Layout-Analysis

This repo contains our (Team: Krusty Krab) codes for DLS2 Document-Layout-Analysis.<br>
Huggingface demo: <a href="https://huggingface.co/spaces/qoobeeshy/yolo-document-layout-analysis">Document Layout Analysis </a><br>
Solution paper: <a href="https://arxiv.org/abs/2309.00848"> A Post-processing Based Bengali Document Layout Analysis With YOLOv8</a><br>
Presentation Slide: <a href="https://www.canva.com/design/DAFrLGganQ8/KFJzZNnj0Ibowk9WoJHyFQ/edit?utm_content=DAFrLGganQ8&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton"> Bengali DLA: A YOLOv8 Based Approach</a><br>
The repository is structured into three folders:<br>
<br>
<b>Document-Layout-Analysis <b><br>
│   README.md   <br>
│<br>
└──Detectron2 <br>
│    &emsp;│   dit-eval.ipynb <br>
│    &emsp;│   dit-sub-template.ipynb <br>
│    &emsp;│   . . .<br>
│    <br>
└──YOLO <br>
│    &emsp;│   publytrain.py <br>
│    &emsp;│   traintbl.py <br>
│    &emsp;│   . . .


* **Detectron2:** This folder contains some of the Detectron2 scripts used for training various models/ evaluation/ inference.
* **YOLO:** This folder contains all the training scripts used to train different YOLOv8 models.

***This repo does not contain any of the model weights. Each folder has its own readme with instructions on how to run the codes.
