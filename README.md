
# Intelligent Drowsiness Detection System
ระบบตรวจจับความง่วงอัจฉริยะ (Intelligent Drowsiness Detection System) ผ่านกล้องเว็บแคมแบบ Real-time พัฒนาด้วย **Next.js**, **OpenCV.js** และ **ONNX Runtime Web** ระบบจะทำการประมวลผลบนเบราว์เซอร์ (Client-side) ทั้งหมด โดยเริ่มจากการตรวจจับใบหน้า ค้นหาดวงตา และใช้โมเดล Deep Learning (YOLO/ONNX) เพื่อวิเคราะห์สถานะของดวงตาว่า "ลืมตา (Awake)" หรือ "หลับตา/ง่วงนอน (Sleepy)" พร้อมระบบแจ้งเตือนด้วยเสียงอัตโนมัติ
##  Features (คุณสมบัติหลัก)
* **Real-time Detection:** ตรวจจับใบหน้าและดวงตาแบบเรียลไทม์ผ่านกล้องเว็บแคม
* **In-Browser Inference:** รันโมเดล AI บนเบราว์เซอร์โดยตรงผ่าน ONNX Runtime Web (WASM) ไม่ต้องส่งข้อมูลภาพขึ้นเซิร์ฟเวอร์ (Privacy-first)
* **Audio Alerts:** มีระบบเล่นเสียงแจ้งเตือนแบบสุ่ม (Sleepy sounds) เมื่อตรวจพบสถานะง่วงนอนติดต่อกัน และหยุดเสียงอัตโนมัติเมื่อกลับมาตื่นตัว
* **High Performance:** ใช้ Haar Cascades ของ OpenCV ในการตีกรอบใบหน้าก่อน เพื่อจำกัดบริเวณการค้นหาดวงตา (ROI) ทำให้ประมวลผลได้รวดเร็วขึ้น
* **Modern UI:** ออกแบบหน้าจอด้วย Tailwind CSS รองรับการแสดงผลเปอร์เซ็นต์ความแม่นยำ (Confidence) และสถานะของระบบแบบเรียลไทม์
## Tech Stack (เทคโนโลยีที่ใช้)
* **Framework:** Next.js (React)
* **Computer Vision:
* ** OpenCV.js (`haarcascade`)
* **Machine Learning:** ONNX Runtime Web (`onnxruntime-web`)
* **Styling:** Tailwind CSS

## Team Diao koy tang
| **Student ID** | **Name**               |
|-----------------|------------------------|
| 67023008        | Apinya Sanghong        |
| 67025077        | Supharoke Roopkhamdee     |
| 67026427        | Pattarawin Rungpanarat |

## Demo
[Demo Intelligent Drowsiness Detection System](https://intelligent-drowsiness-detection-system.vercel.app/)


## Contact
**หากมีคำถาม สามารถติดต่อผู้ดูแลโปรเจค:**
  -  อีเมล: naysasatadur5555@gmail.com
  -  GitHub: [https://github.com/pxttxrxwxn](https://github.com/pxttxrxwxn)