# Demo narration (read in one go, ~6 min 30 s at a calm pace)

Record it as ONE audio file (any common format). The video visuals are cut to these sections, so please keep
the section order and do not skip any. Short pauses between sections are fine; the video is timed to your audio
afterwards. Words in *[brackets]* are stage directions, do not read them. Rewrite freely in your own words, but
keep the facts and numbers.

---

**1. Introduction** *(~35 s)*

Hello, I am Muhammad Musa Khan, roll number 23i-0526. This is my Generative AI Assignment 2: TrialMatch, a
criterion-level clinical trial screening system. Given a patient note and one trial criterion, it returns an
eligibility label and the sentences that support it. It is an educational prototype, not a clinical enrolment
tool. Under the hood there is a LoRA-tuned Qwen2.5 1.5 billion parameter model, and I implemented the KV cache,
tiled attention and speculative decoding myself.

**2. Repository and data** *(~40 s)*

The repository has the source code, the browser app, the results and logs, and the IEEE paper. I used only the
TrialGPT criterion annotations dataset. The 53 patients were shuffled with NumPy seed 42 and split 37, 8 and 8
for train, validation and test, so every patient is in exactly one split. The split manifest is saved in the
results folder, and all GPT-4 fields were removed before training. The test set was evaluated once, at the end.

**3. Screening** *(~95 s)*

This is the app, running from the command line, so no notebook is needed. The header shows the models and the GPU. I load the synthetic sample and press run. For every criterion we see the label, the cited sentence numbers with their text, and an explanation built only from those sentences. Each criterion takes about two thirds of a second on the T4. The angina criterion is labelled included, supported by sentence zero. The diabetes criterion is also labelled included, which is right, but look at the evidence: the model cited sentence one, about the chest pain, while the diabetes is in sentence two. So the label is correct and the citation is wrong, and showing the evidence is what lets a reviewer catch that. For the exclusion criteria the model says not excluded without citing anything, and the app flags those as unsupported. At the top is the overall outcome, from four simple rules in the About tab: any excluded or not-included criterion makes the patient likely ineligible, missing information makes the outcome undetermined, and only when everything is satisfied is the patient preliminarily eligible. This is an aggregation of criterion labels, not ground truth.

**4. Input validation** *(~35 s)*

The app handles bad input. If I clear the note and run, I get a clear error message instead of a crash, and the server keeps working. The same happens for criteria with an invalid type or almost no text. I can also paste a completely new note and new criteria, which is how an unseen case would be tested.

**5. Report export** *(~30 s)*

After a run I can download the screening report as HTML or JSON. It contains the note, each decision with its
evidence, the overall outcome, and the provenance: the models, the adapter checksum, the decoding mode and the
time, together with the disclaimer that this is not a clinical decision.

**6. Inference lab** *(~65 s)*

The Inference lab runs the same criteria through three decoders. First, no cache, where every new token recomputes the whole sequence. Second, my KV-cache decoder, which processes the prompt once and then one token per step, reusing stored keys and values; for this model that is 28 kilobytes per token. Third, speculative decoding, where a fine-tuned 0.5 billion parameter draft proposes tokens and the 1.5 billion target verifies them in one pass. Live on the T4, for these five criteria, no cache took 4.2 seconds, the cache 2.5 seconds, and speculative decoding 3.1 seconds with 90 percent of the draft tokens accepted. In all three modes the tokens are identical. So the cache clearly helps, and speculative decoding is correct but slower than plain caching. The paper measures this carefully: the cache is 1.3 to 9 times faster depending on prompt length, and speculative decoding runs at about 0.8 to 0.9 times the cached speed, because in my PyTorch engine the small draft costs the same 43 milliseconds per token as the large model.

**7. Attention lab** *(~35 s)*

The Attention lab compares three implementations: the reference attention, my tiled online-softmax attention, and PyTorch's fused kernel, all checked against a float64 computation. At sequence length 2048 in half precision, all three have the same error, about 7.7 times ten to the minus four, which is just fp16 rounding. In float32, in the paper, the error is about 1.6 times ten to the minus five. The tiled version never builds the full score matrix: here it used 16 megabytes against 384 for the reference, and in the paper at length 4096 it used 51 megabytes against 2.3 gigabytes. But it is much slower in pure PyTorch, 298 milliseconds against 13 for the reference and half a millisecond for the fused kernel, because every tile is a separate kernel launch.

**8. Experiments and results** *(~40 s)*

The Experiments tab shows the saved results. On the held-out test set, 206 criteria from 8 patients, the base
model reaches 12 percent accuracy. After LoRA fine-tuning it reaches 72 percent accuracy, macro-F1 0.69, and
evidence F1 0.56, compared with a majority-label baseline of 63 percent accuracy and macro-F1 0.27. Inclusion
criteria are easier than exclusion criteria. The main error is saying a criterion is not met when the note is
simply silent.

**9. Limitations and close** *(~30 s)*

The limitations are important. The test set is only 8 patients, so rare classes rest on a few examples. I ran
one training run per setting, and all timings come from one T4 GPU. I also found that Hugging Face's default
generation settings add a repetition penalty, which I report separately in the paper. Thank you for watching.
