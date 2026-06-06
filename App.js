import React, { useMemo, useRef, useState } from 'react';
import { Alert, SafeAreaView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { StatusBar } from 'expo-status-bar';
import { WebView } from 'react-native-webview';

const html = String.raw`
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
  <style>
    html, body { margin: 0; padding: 0; background: #f6f8fb; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    #wrap { padding: 12px; }
    #status { color: #334155; font-size: 13px; line-height: 18px; margin-bottom: 10px; }
    #canvas { width: 100%; border-radius: 8px; background: #e2e8f0; display: block; }
    .hint { color: #64748b; font-size: 12px; line-height: 17px; margin-top: 10px; }
  </style>
</head>
<body>
  <div id="wrap">
    <div id="status">Loading OpenCV.js...</div>
    <canvas id="canvas"></canvas>
    <div class="hint">Choose an image from the native controls above. Processing happens inside this OpenCV canvas.</div>
  </div>
  <script async src="https://docs.opencv.org/4.x/opencv.js" onload="cvLoaded()"></script>
  <script>
    let cvReady = false;
    let cascadeReady = false;
    const cascadeFiles = [
      'haarcascade_frontalface_default.xml',
      'haarcascade_frontalface_alt2.xml',
      'haarcascade_eye_tree_eyeglasses.xml'
    ];
    const statusEl = document.getElementById('status');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

    function post(type, payload) {
      window.ReactNativeWebView && window.ReactNativeWebView.postMessage(JSON.stringify({ type, payload }));
    }

    function setStatus(text) {
      statusEl.textContent = text;
      post('status', text);
    }

    function cvLoaded() {
      cv.onRuntimeInitialized = async () => {
        cvReady = true;
        setStatus('OpenCV.js is ready. Loading face detector...');
        try {
          for (const file of cascadeFiles) {
            const url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/' + file;
            const response = await fetch(url);
            const data = new Uint8Array(await response.arrayBuffer());
            cv.FS_createDataFile('/', file, data, true, false, false);
          }
          cascadeReady = true;
          setStatus('OpenCV is ready. Pick a photo to begin.');
        } catch (error) {
          setStatus('OpenCV is ready, but the face model could not be downloaded.');
        }
      };
    }

    function drawImage(base64, callback) {
      const image = new Image();
      image.onload = () => {
        const maxWidth = 1100;
        const scale = Math.min(1, maxWidth / image.width);
        canvas.width = Math.round(image.width * scale);
        canvas.height = Math.round(image.height * scale);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        callback();
      };
      image.onerror = () => setStatus('Could not load the selected image.');
      image.src = 'data:image/jpeg;base64,' + base64;
    }

    function detectFace(base64) {
      if (!cvReady) {
        setStatus('OpenCV is still loading. Try again in a moment.');
        return;
      }
      if (!cascadeReady) {
        setStatus('Face detector model is not ready. Check internet connection and reload.');
        return;
      }
      drawImage(base64, () => {
        const src = cv.imread(canvas);
        const gray = new cv.Mat();
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
        cv.equalizeHist(gray, gray);
        const candidates = [];
        const minFace = Math.round(Math.min(canvas.width, canvas.height) * 0.18);

        function collectFaces(file, scaleFactor, neighbors, minSize) {
          const found = new cv.RectVector();
          const classifier = new cv.CascadeClassifier();
          classifier.load(file);
          classifier.detectMultiScale(gray, found, scaleFactor, neighbors, 0, new cv.Size(minSize, minSize));
          for (let i = 0; i < found.size(); i += 1) {
            const face = found.get(i);
            candidates.push({ x: face.x, y: face.y, width: face.width, height: face.height, source: file });
          }
          found.delete();
          classifier.delete();
        }

        collectFaces('haarcascade_frontalface_default.xml', 1.06, 3, Math.max(70, minFace));
        collectFaces('haarcascade_frontalface_alt2.xml', 1.05, 3, Math.max(70, minFace));

        let boxes = mergeBoxes(candidates.filter((box) => box.width >= minFace && box.height >= minFace));
        if (!boxes.length) {
          boxes = estimateFromEyes(gray);
        }
        if (!boxes.length && candidates.length) {
          boxes = [expandSmallCandidate(bestCandidate(candidates))];
        }

        ctx.lineWidth = Math.max(3, Math.round(canvas.width / 150));
        ctx.strokeStyle = '#10b981';
        ctx.fillStyle = 'rgba(16, 185, 129, 0.18)';
        boxes.forEach((face) => {
          ctx.fillRect(face.x, face.y, face.width, face.height);
          ctx.strokeRect(face.x, face.y, face.width, face.height);
        });
        const count = boxes.length;
        setStatus(count ? 'Detected ' + count + ' face(s).' : 'No full face detected. Try a brighter front-facing photo without heavy helmet/glare.');
        src.delete(); gray.delete();
      });
    }

    function overlap(a, b) {
      const x1 = Math.max(a.x, b.x);
      const y1 = Math.max(a.y, b.y);
      const x2 = Math.min(a.x + a.width, b.x + b.width);
      const y2 = Math.min(a.y + a.height, b.y + b.height);
      const area = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
      const minArea = Math.min(a.width * a.height, b.width * b.height);
      return minArea ? area / minArea : 0;
    }

    function mergeBoxes(boxes) {
      const sorted = boxes
        .filter((box) => box.width > 0 && box.height > 0)
        .sort((a, b) => (b.width * b.height) - (a.width * a.height));
      const merged = [];
      for (const box of sorted) {
        if (!merged.some((item) => overlap(item, box) > 0.45)) {
          merged.push(clampBox(box));
        }
      }
      return merged.slice(0, 3);
    }

    function bestCandidate(candidates) {
      const centerX = canvas.width / 2;
      return candidates
        .slice()
        .sort((a, b) => {
          const aScore = a.width * a.height - Math.abs((a.x + a.width / 2) - centerX) * 20;
          const bScore = b.width * b.height - Math.abs((b.x + b.width / 2) - centerX) * 20;
          return bScore - aScore;
        })[0];
    }

    function expandSmallCandidate(box) {
      const centerX = box.x + box.width / 2;
      const centerY = box.y + box.height / 2;
      const size = Math.min(Math.min(canvas.width, canvas.height) * 0.82, Math.max(box.width, box.height) * 4.1);
      return clampBox({
        x: centerX - size / 2,
        y: centerY - size * 0.62,
        width: size,
        height: size * 1.06
      });
    }

    function estimateFromEyes(gray) {
      const eyes = [];
      const vector = new cv.RectVector();
      const eyeClassifier = new cv.CascadeClassifier();
      eyeClassifier.load('haarcascade_eye_tree_eyeglasses.xml');
      eyeClassifier.detectMultiScale(gray, vector, 1.06, 3, 0, new cv.Size(24, 24));
      for (let i = 0; i < vector.size(); i += 1) {
        const eye = vector.get(i);
        eyes.push({ x: eye.x, y: eye.y, width: eye.width, height: eye.height });
      }
      vector.delete();
      eyeClassifier.delete();

      let bestPair = null;
      let bestScore = -Infinity;
      for (let i = 0; i < eyes.length; i += 1) {
        for (let j = i + 1; j < eyes.length; j += 1) {
          const a = eyes[i];
          const b = eyes[j];
          const ax = a.x + a.width / 2;
          const bx = b.x + b.width / 2;
          const ay = a.y + a.height / 2;
          const by = b.y + b.height / 2;
          const dx = Math.abs(ax - bx);
          const dy = Math.abs(ay - by);
          const yOk = dy < Math.max(a.height, b.height) * 0.9;
          const distanceOk = dx > Math.min(canvas.width, canvas.height) * 0.08 && dx < Math.min(canvas.width, canvas.height) * 0.42;
          if (!yOk || !distanceOk) continue;
          const score = dx * 10 - dy * 6 + Math.min(a.width * a.height, b.width * b.height);
          if (score > bestScore) {
            bestScore = score;
            bestPair = { left: ax < bx ? a : b, right: ax < bx ? b : a };
          }
        }
      }

      if (!bestPair) return [];
      const leftX = bestPair.left.x + bestPair.left.width / 2;
      const rightX = bestPair.right.x + bestPair.right.width / 2;
      const eyeY = ((bestPair.left.y + bestPair.left.height / 2) + (bestPair.right.y + bestPair.right.height / 2)) / 2;
      const eyeDistance = rightX - leftX;
      const size = eyeDistance * 2.45;
      const centerX = (leftX + rightX) / 2;
      return [clampBox({
        x: centerX - size / 2,
        y: eyeY - size * 0.38,
        width: size,
        height: size * 1.18
      })];
    }

    function clampBox(box) {
      const x = Math.max(0, Math.round(box.x));
      const y = Math.max(0, Math.round(box.y));
      const width = Math.min(canvas.width - x, Math.round(box.width));
      const height = Math.min(canvas.height - y, Math.round(box.height));
      return { x, y, width, height };
    }

    function removeShadow(base64) {
      if (!cvReady) {
        setStatus('OpenCV is still loading. Try again in a moment.');
        return;
      }
      drawImage(base64, () => {
        const src = cv.imread(canvas);
        const rgb = new cv.Mat();
        cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);
        const channels = new cv.MatVector();
        cv.split(rgb, channels);
        const resultChannels = new cv.MatVector();
        const kernel = cv.Mat.ones(7, 7, cv.CV_8U);

        for (let i = 0; i < channels.size(); i += 1) {
          const plane = channels.get(i);
          const dilated = new cv.Mat();
          const background = new cv.Mat();
          const diff = new cv.Mat();
          const normalized = new cv.Mat();
          cv.dilate(plane, dilated, kernel);
          cv.medianBlur(dilated, background, 31);
          cv.absdiff(plane, background, diff);
          cv.bitwise_not(diff, diff);
          cv.normalize(diff, normalized, 0, 255, cv.NORM_MINMAX);
          resultChannels.push_back(normalized);
          plane.delete(); dilated.delete(); background.delete(); diff.delete();
        }

        const result = new cv.Mat();
        cv.merge(resultChannels, result);
        cv.imshow(canvas, result);
        setStatus('Shadow removal completed. The page should look flatter and brighter.');
        src.delete(); rgb.delete(); channels.delete(); resultChannels.delete(); kernel.delete(); result.delete();
      });
    }

    document.addEventListener('message', function(event) {
      handleMessage(event.data);
    });
    window.addEventListener('message', function(event) {
      handleMessage(event.data);
    });

    function handleMessage(raw) {
      try {
        const message = JSON.parse(raw);
        if (message.mode === 'face') detectFace(message.base64);
        if (message.mode === 'shadow') removeShadow(message.base64);
      } catch (error) {
        setStatus('Invalid message from React Native.');
      }
    }
  </script>
</body>
</html>
`;

export default function App() {
  const webViewRef = useRef(null);
  const [mode, setMode] = useState('face');
  const [status, setStatus] = useState('OpenCV canvas is starting...');

  const modeText = useMemo(() => (
    mode === 'face'
      ? 'Pick a front-facing photo. The app draws green boxes around detected faces.'
      : 'Pick a book/page photo with shadow. The app applies an OpenCV shadow-removal pipeline.'
  ), [mode]);

  const pickImage = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permission needed', 'Please allow photo access to select an image.');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      base64: true,
      quality: 0.92
    });

    if (result.canceled || !result.assets?.[0]?.base64) return;
    setStatus(mode === 'face' ? 'Running face detection...' : 'Removing shadow...');
    const payload = JSON.stringify({ mode, base64: result.assets[0].base64 });
    webViewRef.current?.postMessage(payload);
  };

  const onMessage = (event) => {
    try {
      const message = JSON.parse(event.nativeEvent.data);
      if (message.type === 'status') setStatus(message.payload);
    } catch (error) {
      setStatus(event.nativeEvent.data);
    }
  };

  return (
    <SafeAreaView style={styles.screen}>
      <StatusBar style="dark" />
      <View style={styles.header}>
        <Text style={styles.kicker}>Mobile Development - Lab 6</Text>
        <Text style={styles.title}>OpenCV Vision Lab</Text>
        <Text style={styles.subtitle}>{modeText}</Text>
      </View>

      <View style={styles.segment}>
        <TouchableOpacity style={[styles.segmentButton, mode === 'face' && styles.segmentOn]} onPress={() => setMode('face')}>
          <Text style={[styles.segmentText, mode === 'face' && styles.segmentTextOn]}>Face Detection</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.segmentButton, mode === 'shadow' && styles.segmentOn]} onPress={() => setMode('shadow')}>
          <Text style={[styles.segmentText, mode === 'shadow' && styles.segmentTextOn]}>Shadow Removal</Text>
        </TouchableOpacity>
      </View>

      <TouchableOpacity style={styles.primaryButton} onPress={pickImage}>
        <Text style={styles.primaryText}>Choose Photo</Text>
      </TouchableOpacity>

      <Text style={styles.status}>{status}</Text>

      <View style={styles.webFrame}>
        <WebView
          ref={webViewRef}
          source={{ html }}
          onMessage={onMessage}
          originWhitelist={['*']}
          javaScriptEnabled
          domStorageEnabled
          mixedContentMode="always"
          allowsInlineMediaPlayback
          style={styles.webView}
        />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: '#edf2f7' },
  header: { paddingHorizontal: 18, paddingTop: 18, paddingBottom: 12 },
  kicker: { color: '#2563eb', fontSize: 12, fontWeight: '900', textTransform: 'uppercase' },
  title: { color: '#111827', fontSize: 30, fontWeight: '900', marginTop: 6 },
  subtitle: { color: '#526173', fontSize: 15, lineHeight: 21, marginTop: 6 },
  segment: { flexDirection: 'row', marginHorizontal: 18, backgroundColor: '#d9e2ec', borderRadius: 8, padding: 4 },
  segmentButton: { flex: 1, minHeight: 42, borderRadius: 7, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 8 },
  segmentOn: { backgroundColor: '#ffffff' },
  segmentText: { color: '#526173', fontWeight: '900', fontSize: 13, textAlign: 'center' },
  segmentTextOn: { color: '#111827' },
  primaryButton: { marginHorizontal: 18, marginTop: 12, minHeight: 50, borderRadius: 8, backgroundColor: '#2563eb', alignItems: 'center', justifyContent: 'center' },
  primaryText: { color: '#ffffff', fontSize: 16, fontWeight: '900' },
  status: { color: '#334155', fontSize: 13, lineHeight: 18, marginHorizontal: 18, marginTop: 10, minHeight: 36 },
  webFrame: { flex: 1, margin: 18, marginTop: 10, borderRadius: 8, overflow: 'hidden', backgroundColor: '#ffffff', borderWidth: 1, borderColor: '#d5dde8' },
  webView: { flex: 1, backgroundColor: '#ffffff' }
});
