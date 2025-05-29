import seaborn as sns
import matplotlib.pyplot as plt
import os

def show_results(results_df, save_path="plots"):
    print("\n===== Performance Comparison =====")
    print(results_df.sort_values(by="R2", ascending=False))

    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=results_df[results_df["Model"] == "KNN"],
        x="K", y="R2", hue="Split"
    )
    plt.title("KNN R² Score by K and Split")
    plt.tight_layout()

    plot_filename = os.path.join(save_path, "knn_r2_comparison.png")
    plt.savefig(plot_filename)
    print(f"\n📁 Plot berhasil disimpan di: {plot_filename}")

    plt.show()