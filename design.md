# UI/UX Design Specification: Minimalist Light Theme

This document defines the visual layout, typography, colors, and engineering strategies for the redesigned Tech Stack Recommender dashboard.

---

## 🎨 1. Visual Aesthetics & Design System

The interface shifts from a dark, decorative theme to a **sleek, minimalist, high-contrast light theme** designed to feel like a modern, professional developer utility rather than an AI chatbot.

### A. Color Palette (Colorful Minimalist Tokens)
```css
:root {
  /* Light Mint-Green to Sky-Blue Linear Backdrop */
  background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 50%, #ecfeff 100%);
  background-attachment: fixed;

  --bg-secondary:    #ffffff; /* Pure white panels */
  --bg-tertiary:     #f1f5f9; /* Soft grey highlights */
  
  --border-clean:    #e2e8f0; /* Fine slate border */
  --border-focus:    #3b82f6; /* Clean royal blue focus */
  
  --text-main:       #0f172a; /* Deep charcoal (high readability) */
  --text-muted:      #64748b; /* Soft slate grey for meta */
  --text-link:       #2563eb; /* Royal blue for actions */
  
  /* MNC Gradients & Glows */
  --grad-primary:    linear-gradient(135deg, #2563eb 0%, #4f46e5 50%, #06b6d4 100%);
  --grad-accent:     linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%);
  --shadow-glow:     0 4px 14px 0 rgba(37, 99, 235, 0.18);
  --shadow-glow-h:   0 8px 24px 0 rgba(37, 99, 235, 0.3);
}
```

### B. Typography
* **Primary / Heading Font:** `Plus Jakarta Sans`, sans-serif (clean, professional geometry)
* **Body Font:** `Inter`, system-ui, sans-serif (optimal for data-heavy reading)
* **Code & Vectors:** `JetBrains Mono`, monospace (crisp weights for numerical arrays)

### C. Iconography
* **Rule:** Zero emojis.
* **Implementation:** Clean, lightweight SVG path icons embedded directly in the HTML markup.
* **Weight:** 1.5px stroke width, monochrome palette matching the text state.

### D. Advanced Micro-interactions & Animations
* **Spring Transitions:** Standard animations use a physics-based spring curve: `200ms cubic-bezier(0.16, 1, 0.3, 1)` for highly responsive layouts.
* **Slide-Up Card Entrance:** Results enter with a spring-loaded `slide-up-pop` keyframe animation.
* **Gradient-Border Highlighting:** The `#1` recommendation card has a custom dual-layered gradient border outline (`linear-gradient(#ffffff, #ffffff) padding-box, var(--grad-primary) border-box`) and matching glow shadow.
* **Active Tab Slide:** Tab selections feature a sliding horizontal underline that scales on the active link.
* **Floating Dot Pulse:** Status markers feature a soft infinite breathing animation.

---

## 📐 2. User Interface Layout & Layout Structure

The layout is split into two primary panels focusing on the core functional use-case:

1. **Workspace Container:**
   * Left Column: **Skill Input Console**. Modern tags field with real-time autocompletion, Top-N slider, and Verbose toggle.
   * Right Column: **Ranked Career Recommendations**. Displays role match scores with sleek linear indicators showing similarity weight.
2. **Supplementary Tabs (System Reference):**
   * Placed in a clean top horizontal bar (Recommender, Worked Example, Dataset Corpus, Verification).
   * Blends with the background when inactive, emphasizing the main utility.

---

## ⚡ 3. Scalability Optimizations (100s of Skills & Roles)

To support expanding the database to hundreds of programming languages, libraries, framework tags, and role categories, we implement the following:

### A. Autocomplete Suggester (Prefix/Substring Lookup)
* Typing in the input box dynamically filters vocabulary terms to show matching skills.
* Prevents users from entering misspelled tags that would trigger out-of-vocabulary (OOV) mismatches.
* Dropdown list is limited to the Top-6 matches to prevent DOM overflow.

### B. Debounced Filtering
* Searching in the Job Dataset filters card structures using a debounced search input (`100ms`).
* Minimizes layout thrashing.

### C. Clean tag management
* Tags are rendered as compact badges with simple `x` close actions.
* The container overflows scrollably rather than expanding the page layout vertically.

### D. Efficient Similarity Loops
* Math calculations convert TypedArrays once and use fast pre-computed indices.
