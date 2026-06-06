# LAB06 - OpenCV Mobile Studio

An Expo Go computer-vision application for Lab 06 Mobile Development. The app provides a polished mobile studio interface for two OpenCV exercises: face detection and book shadow removal.

## Lab Requirements

- Review and apply the OpenCV Android SDK installation concept.
- Implement a face detection application using a personal face photo.
- Implement a shadow-removal function for a favorite book/page image.
- Provide a demo-ready mobile interface with clear visual output.

## Features

- Professional OpenCV studio-style interface.
- Requirement checklist based on the Lab 06 PDF.
- Face Detection module using OpenCV Haar Cascade models.
- Improved face-box estimation for difficult selfies with glasses, helmets, or partial detections.
- Book Shadow Removal module using OpenCV image processing.
- Native photo picker with gallery permission handling.
- Output canvas rendered inside a WebView with OpenCV.js.
- Expo SDK 54 support for the current Expo Go app.

## Computer Vision Pipeline

### Face Detection

The face module loads multiple OpenCV Haar Cascade models:

- `haarcascade_frontalface_default.xml`
- `haarcascade_frontalface_alt2.xml`
- `haarcascade_eye_tree_eyeglasses.xml`

The app detects candidate face regions, filters small false positives, merges overlapping boxes, and uses eye-based estimation as a fallback when the face is partially covered.

### Shadow Removal

The shadow-removal module processes each RGB channel with:

- dilation
- median background estimation
- absolute difference
- inverse intensity mapping
- normalization

This produces a brighter, flatter page image suitable for book/page photos with shadows.

## Tech Stack

- React Native
- Expo Go SDK 54
- Expo Image Picker
- React Native WebView
- OpenCV.js

## Run The Project

```bash
npm install
npx expo start -c
```

Scan the QR code with Expo Go.

## Demo Guide

1. Wait until the status says OpenCV is ready.
2. Choose `Face Detection`.
3. Tap `Select Face Photo` and choose a front-facing photo.
4. Switch to `Book Shadow Removal`.
5. Tap `Select Book Photo` and choose a page/book image with visible shadow.
6. Review the processed result in the OpenCV output canvas.

## Expo Go Note

The original Lab 06 references native Android OpenCV SDK. Expo Go cannot load native Android OpenCV modules directly, so this project uses OpenCV.js inside a WebView. The implementation still demonstrates the required OpenCV concepts and produces visible computer-vision results on a mobile device without requiring a custom native build.
