# SWE015 submission

- **Project** Handwritten Answer Digitizer, Use Case 1
- **Student** Abdalraouf Alkabir
- **Student ID** 2309015482

## Files and links

- [Submission ZIP](https://github.com/afel6/swe015-handwritten-digitizer/releases/download/v1.0.1-submission/SWE015_Submission_2309015482.zip) contains the source, tests, results, decks, and offline demo.
- [Presentation PDF](SWE015_Presentation.pdf) and [PowerPoint](SWE015_Presentation.pptx)
- [Official two-minute demo on YouTube](https://youtu.be/m2d_cP5pWOk)
- [GitHub repository](https://github.com/afel6/swe015-handwritten-digitizer)
- [Benchmark tables](../results/benchmark_results.md) and [grader instructions](../README.md#for-graders)

## Quick laptop check

Follow the [Windows or Linux setup](../README.md#download-and-run) using Python 3.12 or 3.13.
The first setup downloads the model files. After the app starts, open
[http://127.0.0.1:8000/](http://127.0.0.1:8000/), select `test_162` in **Answer reader**, and click
**Recognize**. The expected suggestion is **EXPELLED**. To check the saved benchmark figures and
model files, run `python verify.py` in the project environment.

The boxed-answer reader runs on the CPU. The optional **Full-page OCR** tab uses Ollama with local
Qwen3-VL. The course benchmark measures generated boxed strips.
