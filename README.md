# 🎨 Paint-by-Numbers Generator
### Turn any photo into a printable, professional-grade coloring book!

![App Preview](assets/preview.png)  
*(Upload any image → get a clean, numbered outline & color swatch just like commercial Paint-by-Number kits.)*

---

## 🖌️ Overview
**Paint-by-Numbers Generator** is a Streamlit web app that uses **AI-powered image segmentation** and **K-Means clustering** to transform any photo into a clean, printable paint-by-numbers coloring page — complete with **numbered outlines**, **color keys**, and a **ready-to-print PDF kit**.

Perfect for artists, hobbyists, teachers, and coloring enthusiasts.

---

## ✨ Features

- 🖼️ **Upload any JPG or PNG**
- 🎨 **Adjust color detail** — Choose from 12 to 64 colors
- 🧹 **Smart region cleanup** — Automatically merges tiny or similar color areas
- ✏️ **Clean, bold outlines** — Edge-aware contour generation
- 🔢 **Automatic numbering** — Each region gets a number for easy painting
- 📄 **Professional PDF output** — Includes a numbered page + color swatch key
- 📦 **Download ZIP Kit** — Get your PBN image, color key, and PDF all in one

---

## 🧠 How It Works

1. **K-Means Color Quantization**  
   Reduces the image to a user-defined number of colors (12–64) using `scikit-learn`.

2. **Region Cleaning & Merging**  
   Removes tiny segments and merges similar color regions for smooth outlines.

3. **Edge-Aware Smoothing**  
   Applies OpenCV’s bilateral filtering to retain edges while softening color noise.

4. **Outline Detection**  
   Uses morphological operations + contour tracing to draw clean black outlines.

5. **Region Numbering**  
   Labels each distinct color area and overlays region numbers with `Pillow`.

6. **PDF & ZIP Output**  
   Generates a print-ready PDF using `ReportLab`, plus a downloadable ZIP kit.

---

## 🚀 Demo
Try it live (once deployed):  
👉 [**Streamlit Cloud Demo (Coming Soon)**](#)

Or run locally:

```bash
git clone https://github.com/yourusername/paint-by-numbers-generator.git
cd paint-by-numbers-generator
pip install -r requirements.txt
streamlit run app.py
# Image2PBN
Upload ANY Image to Generate a Printable Paint by Numbers Template.
