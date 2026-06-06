import React, { useMemo, useRef, useState } from 'react';
import { Alert, SafeAreaView, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
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

  const activeTask = useMemo(() => (
    mode === 'face'
      ? {
        label: 'Face Detection',
        accent: '#38bdf8',
        button: 'Select Face Photo',
        objective: 'Detect a face from your own photo using OpenCV Haar Cascade models.',
        output: 'Result draws an assisted face bounding box on the selected image.'
      }
      : {
        label: 'Book Shadow Removal',
        accent: '#f59e0b',
        button: 'Select Book Photo',
        objective: 'Reduce shadows from a favorite book/page photo using OpenCV image processing.',
        output: 'Result normalizes page brightness and flattens heavy shadow regions.'
      }
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
      <StatusBar style="light" />
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.hero}>
          <View style={styles.topLine}>
            <Text style={styles.course}>Mobile Development</Text>
            <View style={styles.labBadge}><Text style={styles.labBadgeText}>Lab 06</Text></View>
          </View>
          <Text style={styles.title}>OpenCV Mobile Studio</Text>
          <Text style={styles.subtitle}>Computer vision workspace for face detection and document shadow correction.</Text>
        </View>

        <View style={styles.requirementsPanel}>
          <Text style={styles.sectionTitle}>PDF Requirements</Text>
          <View style={styles.requirementGrid}>
            <Requirement label="OpenCV SDK" text="Install guide reviewed; Expo demo uses OpenCV.js." />
            <Requirement label="Face Detection" text="Use a personal face photo and draw detection output." />
            <Requirement label="Shadow Removal" text="Use a book/page image with shadow and remove it." />
            <Requirement label="Submission" text="Demo-ready app with clear UI and visual results." />
          </View>
        </View>

        <Text style={styles.sectionTitle}>Vision Modules</Text>
        <View style={styles.taskGrid}>
          <TaskCard
            title="Face Detection"
            code="01"
            active={mode === 'face'}
            accent="#38bdf8"
            text="OpenCV Haar Cascade with fallback face-box estimation for selfies, glasses, and helmets."
            onPress={() => setMode('face')}
          />
          <TaskCard
            title="Shadow Removal"
            code="02"
            active={mode === 'shadow'}
            accent="#f59e0b"
            text="Dilation, median background estimation, inverse difference, and normalization."
            onPress={() => setMode('shadow')}
          />
        </View>

        <View style={styles.workflow}>
          <View style={styles.workflowHeader}>
            <View>
              <Text style={styles.activeLabel}>Active Task</Text>
              <Text style={styles.activeTitle}>{activeTask.label}</Text>
            </View>
            <View style={[styles.statusDot, { backgroundColor: activeTask.accent }]} />
          </View>
          <Text style={styles.workflowText}>{activeTask.objective}</Text>
          <Text style={styles.workflowText}>{activeTask.output}</Text>
          <TouchableOpacity style={[styles.primaryButton, { backgroundColor: activeTask.accent }]} onPress={pickImage}>
            <Text style={styles.primaryText}>{activeTask.button}</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.statusPanel}>
          <Text style={styles.statusTitle}>Processing Status</Text>
          <Text style={styles.status}>{status}</Text>
        </View>

        <View style={styles.outputPanel}>
          <View style={styles.outputHeader}>
            <Text style={styles.outputTitle}>OpenCV Output Canvas</Text>
            <Text style={styles.outputMeta}>Runs inside WebView</Text>
          </View>
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
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function Requirement({ label, text }) {
  return (
    <View style={styles.requirementCard}>
      <Text style={styles.requirementLabel}>{label}</Text>
      <Text style={styles.requirementText}>{text}</Text>
    </View>
  );
}

function TaskCard({ title, code, active, accent, text, onPress }) {
  return (
    <TouchableOpacity style={[styles.taskCard, active && { borderColor: accent }]} activeOpacity={0.86} onPress={onPress}>
      <View style={styles.taskTop}>
        <Text style={[styles.taskCode, { color: accent }]}>{code}</Text>
        <View style={[styles.taskPill, active && { backgroundColor: accent }]}>
          <Text style={[styles.taskPillText, active && styles.taskPillTextOn]}>{active ? 'Selected' : 'Tap'}</Text>
        </View>
      </View>
      <Text style={styles.taskTitle}>{title}</Text>
      <Text style={styles.taskText}>{text}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: '#08111f' },
  content: { paddingHorizontal: 16, paddingTop: 14, paddingBottom: 30 },
  hero: { backgroundColor: '#0f1c2f', borderRadius: 8, padding: 18, borderWidth: 1, borderColor: '#1f3658' },
  topLine: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  course: { color: '#93c5fd', fontSize: 12, fontWeight: '900', textTransform: 'uppercase' },
  labBadge: { backgroundColor: '#172554', borderRadius: 8, paddingHorizontal: 10, minHeight: 30, justifyContent: 'center' },
  labBadgeText: { color: '#bfdbfe', fontWeight: '900', fontSize: 12 },
  title: { color: '#ffffff', fontSize: 30, fontWeight: '900', marginTop: 12 },
  subtitle: { color: '#b6c3d4', fontSize: 15, lineHeight: 21, marginTop: 8 },
  requirementsPanel: { marginTop: 14 },
  sectionTitle: { color: '#ffffff', fontSize: 19, fontWeight: '900', marginTop: 16, marginBottom: 10 },
  requirementGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  requirementCard: { width: '48.5%', backgroundColor: '#0f1c2f', borderRadius: 8, borderWidth: 1, borderColor: '#213957', padding: 12, minHeight: 92 },
  requirementLabel: { color: '#67e8f9', fontSize: 13, fontWeight: '900' },
  requirementText: { color: '#b6c3d4', fontSize: 12, lineHeight: 17, marginTop: 6 },
  taskGrid: { gap: 10 },
  taskCard: { backgroundColor: '#101827', borderRadius: 8, borderWidth: 1, borderColor: '#25344a', padding: 14 },
  taskTop: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  taskCode: { fontSize: 18, fontWeight: '900' },
  taskPill: { borderRadius: 8, borderWidth: 1, borderColor: '#334155', paddingHorizontal: 10, minHeight: 28, justifyContent: 'center' },
  taskPillText: { color: '#94a3b8', fontSize: 12, fontWeight: '900' },
  taskPillTextOn: { color: '#07111f' },
  taskTitle: { color: '#ffffff', fontSize: 18, fontWeight: '900', marginTop: 10 },
  taskText: { color: '#aebdd0', fontSize: 13, lineHeight: 19, marginTop: 6 },
  workflow: { backgroundColor: '#f8fafc', borderRadius: 8, padding: 15, marginTop: 14 },
  workflowHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  activeLabel: { color: '#64748b', fontSize: 12, fontWeight: '900', textTransform: 'uppercase' },
  activeTitle: { color: '#0f172a', fontSize: 21, fontWeight: '900', marginTop: 2 },
  statusDot: { width: 16, height: 16, borderRadius: 8 },
  workflowText: { color: '#475569', fontSize: 14, lineHeight: 20, marginTop: 8 },
  primaryButton: { marginTop: 14, minHeight: 50, borderRadius: 8, alignItems: 'center', justifyContent: 'center' },
  primaryText: { color: '#07111f', fontSize: 16, fontWeight: '900' },
  statusPanel: { backgroundColor: '#0f1c2f', borderRadius: 8, borderWidth: 1, borderColor: '#213957', padding: 14, marginTop: 14 },
  statusTitle: { color: '#ffffff', fontSize: 15, fontWeight: '900' },
  status: { color: '#c9d6e5', fontSize: 13, lineHeight: 19, marginTop: 6 },
  outputPanel: { marginTop: 14, backgroundColor: '#0f1c2f', borderRadius: 8, borderWidth: 1, borderColor: '#213957', padding: 12 },
  outputHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 },
  outputTitle: { color: '#ffffff', fontSize: 16, fontWeight: '900' },
  outputMeta: { color: '#94a3b8', fontSize: 12, fontWeight: '800' },
  webFrame: { height: 430, borderRadius: 8, overflow: 'hidden', backgroundColor: '#ffffff', borderWidth: 1, borderColor: '#2f4463' },
  webView: { flex: 1, backgroundColor: '#ffffff' }
});
