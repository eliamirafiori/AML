# Group 3 Project Report

## Semantic Enrichment in Few-Shot Learning: Bridging the Gap between Numeric IDs and Foundation Models

Team: Delucchi Tommaso; Mirafiori Elia; Segalla Filippo; Pettene Vittorio

## Individual Contribution

- Elia Mirafiori: He focused on the theoretical side, the core design of the architecture, and the general coordination of the project. He set up the main guidelines for our pipeline, led the first phase of the project, and managed the testing cycles and workload distribution. He also wrote and structured the final report together with Vittorio.
- Filippo Segalla: He took care of the compute infrastructure, backend development, and hardware setup. He provided the heavy-duty computing power needed for the resource-intensive LLaVA tasks. On top of that, he built the feature-extraction pipelines and integrated the multi-backbone ensemble with the final linear SVM classifier.
- Vittorio Pettene: He was responsible for the evaluation, benchmarking, and validation phases. He handled the main LLaVA testing loops, ran the ablation studies that proved that semantic enrichment works better than standard zero-shot architectures, and managed the high-performance environments. He also worked closely with Elia on writing and compiling this final report.
- Tommaso Delucchi: He played a key role in the testing, architecture, and presentation phases. In the beginning, he ran the initial tests with CLIP under Elia’s guidance. He also came up with the idea of condensing LLaVA’s long captions into a single descriptive keyword to make the text prototypes sharper. Later, he implemented ConvNeXt and Swin Transformer into the pipeline and tested them. Finally, he led the creation of the presentation slides and helped writing this report.
