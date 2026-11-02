# ContextLens

ContextLens is a real-time conversation assistant that provides **gentle, optional** social-cue support by analyzing **speech**, **body language**, and **environmental context**. Built as a lightweight web app (demoed on **Meta Quest 3**), it helps users—especially **introverts and neurodivergent individuals**—navigate social interactions with more clarity and confidence.

---

## Demo
- Devpost: **(https://devpost.com/software/context-lens-for-education)**

---

## Features
- Real-time rolling transcript (few lines at a time)
- Subtle corner popups for social cues (only when needed)
- Speech analysis + transcription (Deepgram)
- Body language + environment understanding (Gemini Vision)
- Multimodal reasoning layer with large context window (Gemini)
- Optional live translation support

---

## Tech Stack
- **Frontend:** Web app (npm)
- **Deployment:** Vercel
- **Speech Transcription:** Deepgram
- **Vision Understanding:** Gemini Vision
- **Reasoning / Brain:** Gemini
- **Audio Capture (optional):** SoX

---

## Getting Started

### Prerequisites
- Node.js 18+
- npm
- API keys:
  - Deepgram
  - Gemini
