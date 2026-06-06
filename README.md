# LAB06 - OpenCV Mobile Demo

This Expo Go app implements the Lab 6 exercises with OpenCV.js inside a mobile WebView.

## Exercises Covered

- Face detection: choose a face photo and detect faces using OpenCV Haar Cascade.
- Shadow removal: choose a book/page photo with shadow and reduce the shadow using OpenCV image-processing steps.

## Run

```bash
npm install
npx expo start -c
```

Scan the QR code with Expo Go.

## Note

The original Lab 6 references Android native OpenCV SDK. Expo Go cannot load native OpenCV Android modules directly, so this project uses OpenCV.js in a WebView to demo the same computer vision concepts on a phone without creating a custom native build.
