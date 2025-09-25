<div align="center">
<h2>🔎FAQ about synthetic video detection model checkpoint</h2>

[Shuqiao Liang](https://github.com/xigua7105)¹, [Jian Liu]()¹, [RenZhang Chen]()²*, [Quanlong Guan]()³*,

¹ School of Intelligent Systems Science and Engineering, Jinan University  
² Modern Educational Technology Center, Jinan University  
³ College of Information Science and Technology, Jinan University

[![license](https://img.shields.io/badge/license-Apache_2.0-blue)](../../LICENSE)

</div>


- <h4>🤔Q1: How to obtain the dataset for training?</h4> ✅A1: The training data only contains ProGan-4cls, and no additional data is needed.
- <h4>🤔Q2: What is the architecture of the model?</h4> ✅A2: The backbone is FerretNet-B (ferretnet-b-median-3) without any adjustments.</h4>
- <h4>🤔Q3: Can the model be used for synthetic image detection?</h4> ✅A3: This model can detect both synthetic images in lossless PNG format and JPEG-compressed lossy images. It exhibits excellent robustness and generalization capabilities, making it suitable for the rapid detection of images and videos posted on social media.
- <h4>🤔Q4: Why does the model trained with the provided code fail to detect synthetic videos or compressed lossy images?</h4> ✅A4: The training data remains the same, while the training process has been optimized. The improved method significantly enhances the robustness. More implementation details will be released soon.