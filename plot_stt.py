import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser=argparse.ArgumentParser()
parser.add_argument(
    "--filepath",
    type=str,
    help="path to csv file",
    required=True
)

args=parser.parse_args()

CSV_PATH = args.filepath

df = pd.read_csv(CSV_PATH)

summary = (
    df.groupby("model")
      .agg(
          avg_xrt=("x_real_time", "mean"),
          avg_wer=("wer", "mean")
      )
      .reset_index()
)

plt.figure(figsize=(6, 5))

for _, r in summary.iterrows():
    plt.scatter(r["avg_xrt"], r["avg_wer"] * 100)
    plt.text(
        r["avg_xrt"] + 0.01,
        r["avg_wer"] * 100,
        r["model"]
    )

plt.xlabel("ASR Speed (× real-time on Raspberry Pi 4)")
plt.ylabel("Average WER (%)")
plt.title("ASR Performance on Raspberry Pi 4")
plt.grid(True)
plt.tight_layout()
plt.show()
