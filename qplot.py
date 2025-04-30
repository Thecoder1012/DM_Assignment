import matplotlib.pyplot as plt
from PIL import Image

# ==== CONFIGURATION ====
# Path to your real CXR image
real_cxr_path = './inference_out/subset_val/Atelectasis_case0/Atelectasis.00006715_000.png'   # <<== change this to your real image path

# Load real CXR image
real_img = Image.open(real_cxr_path).convert('RGB')

# Set up figure
fig, axs = plt.subplots(2, 2, figsize=(16, 8))
fig.suptitle('Qualitative Results - Chest X-ray Inference', fontsize=24, fontweight='bold')

# Section 1: Real CXR Image
axs[0, 0].imshow(real_img)
axs[0, 0].axis('off')
axs[0, 0].set_title('CXR Image', fontsize=18)

# Section 2: Predictions
axs[0, 1].axis('off')
predictions_text = (
    "Top-3 Predicted Conditions:\n"
    "- Atelectasis: 35.0%\n"
    "- Cardiomegaly: 31.4%\n"
    "- Consolidation: 24.7%"
)
axs[0, 1].text(0.05, 0.5, predictions_text, fontsize=16, va='center', ha='left', wrap=True)

# Section 3: Medical Report
axs[1, 0].axis('off')
medical_report_text = (
    "Medical Report:\n"
    "Findings:\n"
    "- Possible right lower lobe atelectasis.\n"
    "- Mild cardiomegaly cannot be ruled out.\n"
    "- No definite consolidation is identified.\n\n"
    "Impression:\n"
    "Suspected atelectasis. Further imaging or clinical correlation is recommended."
)
axs[1, 0].text(0.05, 0.5, medical_report_text, fontsize=14, va='center', ha='left', wrap=True)

# Section 4: Summary and Simplified Report
axs[1, 1].axis('off')
summary_text = (
    "Summary and Patient-Friendly Explanation:\n"
    "- Possible right lower lobe atelectasis suspected.\n"
    "- Mild cardiomegaly cannot be ruled out.\n"
    "- No consolidation identified.\n"
    "- Further imaging or clinical correlation is recommended.\n\n"
    "Simplified Report:\n"
    "Possible partial collapse of the lung in the lower right side. "
    "Heart may be slightly larger than normal. "
    "No lung infection seen. Further tests recommended."
)
axs[1, 1].text(0.05, 0.5, summary_text, fontsize=14, va='center', ha='left', wrap=True)

# Final layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('qualitative_results_real_cxr.png', dpi=900)
plt.show()
