/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";
import Image from "next/image";

type CvType = any;

export default function Home() {

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [status, setStatus] = useState<string>("ยังไม่เริ่ม");
  const [eyeState, setEyeState] = useState<string>("-");
  const [conf, setConf] = useState<number>(0);


  const cvRef = useRef<CvType | null>(null);
  const faceCascadeRef = useRef<any>(null);
  const eyeCascadeRef = useRef<any>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const classesRef = useRef<string[] | null>(null);

  const lastAudioPlayTimeRef = useRef<number>(0);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);
  const stopAudioTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  async function loadOpenCV() {
    if (typeof window === "undefined") return;

    if ((window as any).cv?.Mat) {
      cvRef.current = (window as any).cv;
      return;
    }

    await new Promise<void>((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/opencv/opencv.js";
      script.async = true;

      script.onload = () => {
        const cv = (window as any).cv;
        if (!cv) return reject(new Error("OpenCV โหลดแล้วแต่ window.cv ไม่มีค่า"));

        const waitReady = () => {
          if ((window as any).cv?.Mat) {
            cvRef.current = (window as any).cv;
            resolve();
          } else {
            setTimeout(waitReady, 50);
          }
        };

        if ("onRuntimeInitialized" in cv) {
          cv.onRuntimeInitialized = () => waitReady();
        } else {
          waitReady();
        }
      };

      script.onerror = () => reject(new Error("โหลด /opencv/opencv.js ไม่สำเร็จ"));
      document.body.appendChild(script);
    });
  }

  async function loadCascades() {
    const cv = cvRef.current;
    if (!cv) throw new Error("cv ยังไม่พร้อม");

    const loadXML = async (url: string, path: string) => {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`โหลด ${url} ไม่สำเร็จ`);
      const data = new Uint8Array(await res.arrayBuffer());
      try {
        cv.FS_unlink(path);
      } catch { }
      cv.FS_createDataFile("/", path, data, true, false, false);
      const classifier = new cv.CascadeClassifier();
      const loaded = classifier.load(path);
      if (!loaded) throw new Error(`โหลด ${path} ไม่สำเร็จ`);
      return classifier;
    };

    faceCascadeRef.current = await loadXML(
      "/opencv/haarcascade_frontalface_default.xml",
      "face.xml"
    );

    eyeCascadeRef.current = await loadXML(
      "/opencv/haarcascade_eye_tree_eyeglasses.xml",
      "eye.xml"
    );
  }

  async function loadModel() {
    const session = await ort.InferenceSession.create(
      "/models/eye_state_yolo.onnx",
      { executionProviders: ["wasm"] }
    );
    sessionRef.current = session;

    try {
      const clsRes = await fetch("/models/classes.json");
      if (clsRes.ok) {
        classesRef.current = await clsRes.json();
      } else {
        classesRef.current = ["Closed", "Open"];
      }
    } catch {
      classesRef.current = ["Closed", "Open"];
    }
  }

  async function startCamera() {
    setStatus("ขอสิทธิ์กล้อง...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
        audio: false,
      });
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setStatus("กำลังทำงาน...");
      requestAnimationFrame(loop);
    } catch (e: any) {
      setStatus(`เปิดกล้องไม่ได้: ${e.message}`);
    }
  }

  function preprocessToTensor(imageCanvas: HTMLCanvasElement) {
    const size = 64;
    const tmp = document.createElement("canvas");
    tmp.width = size;
    tmp.height = size;
    const ctx = tmp.getContext("2d")!;
    ctx.drawImage(imageCanvas, 0, 0, size, size);

    const imgData = ctx.getImageData(0, 0, size, size).data;
    const float = new Float32Array(1 * 3 * size * size);

    let idx = 0;
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < size * size; i++) {
        const r = imgData[i * 4 + 0] / 255.0;
        const g = imgData[i * 4 + 1] / 255.0;
        const b = imgData[i * 4 + 2] / 255.0;
        float[idx++] = c === 0 ? r : c === 1 ? g : b;
      }
    }

    return new ort.Tensor("float32", float, [1, 3, size, size]);
  }

  function softmax(logits: Float32Array) {
    let max = -Infinity;
    for (const v of logits) max = Math.max(max, v);
    const exps = logits.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  }

  function playRandomSleepySound() {
    if (stopAudioTimerRef.current) {
      clearTimeout(stopAudioTimerRef.current);
      stopAudioTimerRef.current = null;
    }

    const now = Date.now();

    if (now - lastAudioPlayTimeRef.current > 3000) {
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
        currentAudioRef.current.currentTime = 0;
      }

      const randomNum = Math.floor(Math.random() * 3) + 1;
      const audio = new Audio(`/voice/sleepy${randomNum}.mp3`);
      
      currentAudioRef.current = audio;
      
      audio.play().catch(err => console.error("Error playing sound:", err));
      lastAudioPlayTimeRef.current = now;
    }
  }

  function handleAwakeState() {
    if (currentAudioRef.current && !currentAudioRef.current.paused && !stopAudioTimerRef.current) {
        stopAudioTimerRef.current = setTimeout(() => {
          if (currentAudioRef.current) {
            currentAudioRef.current.pause();
            currentAudioRef.current.currentTime = 0;
          }
          stopAudioTimerRef.current = null;
        }, 3000);
    }
  }

  function capitalizeFirstLetter(string: string) {
      if (!string) return string;
      return string.charAt(0).toUpperCase() + string.slice(1);
  }

  async function loop() {
    try {
      const cv = cvRef.current;
      const faceCascade = faceCascadeRef.current;
      const eyeCascade = eyeCascadeRef.current;
      const session = sessionRef.current;
      const classes = classesRef.current;

      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!cv || !faceCascade || !eyeCascade || !session || !classes || !video || !canvas) {
        requestAnimationFrame(loop);
        return;
      }

      const ctx = canvas.getContext("2d")!;
      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      ctx.drawImage(video, 0, 0);

      const src = cv.imread(canvas);
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
      const faces = new cv.RectVector();
      const faceMinSize = new cv.Size(100, 100);
      faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, faceMinSize);

      let bestFaceRect: any = null;
      let maxArea = 0;

      for (let i = 0; i < faces.size(); i++) {
        const r = faces.get(i);
        const area = r.width * r.height;
        if (area > maxArea) {
          maxArea = area;
          bestFaceRect = r;
        }
      }

      if (bestFaceRect) {
        ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
        ctx.lineWidth = 2;
        ctx.strokeRect(bestFaceRect.x, bestFaceRect.y, bestFaceRect.width, bestFaceRect.height);

        const roiX = bestFaceRect.x;
        const roiY = bestFaceRect.y + (bestFaceRect.height * 0.15);
        const roiW = bestFaceRect.width;
        const roiH = bestFaceRect.height * 0.40;

        const roiRect = new cv.Rect(roiX, roiY, roiW, roiH);
        const roiGray = gray.roi(roiRect);

        const eyes = new cv.RectVector();
        const eyeMinSize = new cv.Size(20, 20);
        eyeCascade.detectMultiScale(roiGray, eyes, 1.1, 3, 0, eyeMinSize);

        if (eyes.size() > 0) {
          let isSleepy = false;
          let finalDisplayLabel = "-";
          let finalPredConf = 0;

          const maxEyesToProcess = Math.min(eyes.size(), 2);
          for (let i = 0; i < maxEyesToProcess; i++) {
            const e = eyes.get(i);
            const absX = roiX + e.x;
            const absY = roiY + e.y;

            const eyeCanvas = document.createElement("canvas");
            eyeCanvas.width = e.width;
            eyeCanvas.height = e.height;
            const ectx = eyeCanvas.getContext("2d")!;
            ectx.drawImage(canvas, absX, absY, e.width, e.height, 0, 0, e.width, e.height);

            const input = preprocessToTensor(eyeCanvas);
            const feeds: Record<string, ort.Tensor> = {};
            feeds[session.inputNames[0]] = input;

            const results = await session.run(feeds);
            const logits = results[session.outputNames[0]].data as Float32Array;
            const probs = softmax(logits);

            let maxIdx = 0;
            for (let p = 1; p < probs.length; p++) {
              if (probs[p] > probs[maxIdx]) maxIdx = p;
            }

            const rawLabel = classes[maxIdx] ?? `Class ${maxIdx}`;
            const displayLabel = capitalizeFirstLetter(rawLabel);
            const predConf = probs[maxIdx];

            // เช็คว่าถ้าเจอตาที่หลับ ให้เปลี่ยน State หลักเป็นหลับทันที
            if (rawLabel.toLowerCase().includes("sleepy")) {
              isSleepy = true;
              finalDisplayLabel = displayLabel;
              finalPredConf = predConf;
            } else if (!isSleepy) {
              finalDisplayLabel = displayLabel;
              finalPredConf = predConf;
            }

            ctx.strokeStyle = "#00ffff";
            ctx.lineWidth = 2;
            ctx.strokeRect(absX, absY, e.width, e.height);

            ctx.fillStyle = "rgba(0,0,0,0.7)";
            ctx.fillRect(absX, absY - 25, 140, 25);
            ctx.fillStyle = "#00ff00";
            ctx.font = "bold 16px sans-serif";
            ctx.fillText(
              `${displayLabel} ${(predConf * 100).toFixed(0)}%`,
              absX + 5,
              absY - 7
            );
          }

          setEyeState(finalDisplayLabel);
          setConf(finalPredConf);

          if (isSleepy) {
            playRandomSleepySound();
          } else {
            handleAwakeState();
          }
        }

        roiGray.delete();
        eyes.delete();
      }

      src.delete();
      gray.delete();
      faces.delete();

      requestAnimationFrame(loop);
    } catch (e: any) {
      setStatus(`ผิดพลาด: ${e?.message ?? e}`);
    }
  }

  useEffect(() => {
    (async () => {
      try {
        setStatus("กำลังโหลด OpenCV...");
        await loadOpenCV();

        setStatus("กำลังโหลด Haar cascades...");
        await loadCascades();

        setStatus("กำลังโหลดโมเดล ONNX...");
        await loadModel();

        setStatus("พร้อม เริ่มกดปุ่ม Start");
      } catch (e: any) {
        setStatus(`เริ่มต้นไม่สำเร็จ: ${e?.message ?? e}`);
      }
    })();
  }, []);

  return (
    <main className="min-h-screen p-4 md:p-6 space-y-4 md:space-y-2 bg-[#FFF8E1] flex flex-col items-center relative">
      <Image
        src="/image/logo.png"
        alt="Logo"
        width={140}
        height={140}
        className="absolute top-4 left-4 w-16 md:w-35 h-auto md:top-5 md:left-5"
      />

      <h1 className="text-2xl md:text-4xl font-extrabold text-[#BF360C] font-[Montserrat] p-4 pt-16 md:p-10 text-center">
        Intelligent Drowsiness Detection System
      </h1>

      <div className="flex flex-col md:flex-row items-center md:items-end w-full max-w-3xl justify-between space-y-3 md:space-y-0">
        <div className="flex space-x-2 font-[Montserrat]">
          <div className="text-xs md:text-sm bg-[#5D4037] text-white py-2 md:py-3 rounded-2xl px-3 w-35 text-center">
            Result: <b className={eyeState.toLowerCase().includes('open') || eyeState.toLowerCase().includes('awake') ? "text-green-600" : "text-[#F82A2A]"}>{eyeState}</b>
            {" "}
          </div>
          <div className="text-xs md:text-sm bg-[#5D4037] text-white p-2 md:p-3 rounded-2xl">
            Conf: <b>{(conf * 100).toFixed(1)}%</b>
          </div>
        </div>
        <div className="text-sm md:text-base text-black font-[Prompt]">
          สถานะ: {status}
        </div>
      </div>

      <div className="relative w-full max-w-3xl">
        <video ref={videoRef} className="hidden" playsInline muted />
        <canvas
          ref={canvasRef}
          className="w-full rounded-2xl md:rounded-3xl border bg-white shadow-sm"
        />
      </div>

      <p className="text-xs md:text-sm text-[#F82A2A] font-[Prompt] text-center px-2">
        หมายเหตุ: ระบบจะค้นหาใบหน้าก่อน แล้วจึงค้นหาดวงตาเฉพาะในส่วนบนของใบหน้า
      </p>
      
      <button
        className="px-6 py-3 rounded-full md:rounded-4xl bg-[#DF5E10] text-white font-[Mochiy_Pop_P_One] text-sm md:text-base transition-transform active:scale-95"
        onClick={startCamera}
      >
        Start Camera
      </button>
    </main>
  );
}