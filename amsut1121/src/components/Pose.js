import React, { useState, useEffect, useRef } from "react";
import Webcam from "react-webcam";

var imageSrc;

const useNotification = (title, options) => {
  if (!("Notification" in window)) {
    return;
  }

  const fireNotif = () => {
    /* 권한 요청 부분 */
    if (Notification.permission !== "granted") {
      Notification.requestPermission().then((permission) => {
        if (permission === "granted") {
          /* 권한을 요청받고 nofi를 생성해주는 부분 */
          new Notification(title, options);
        } else {
          return;
        }
      });
    } else {
      /* 권한이 있을때 바로 noti 생성해주는 부분 */
      new Notification(title, options);
    }
  };
  return fireNotif;
};

function Pose(pros) {
  const [posture, setPosture] = useState("0");

  useEffect(() => {
    let a;
    if (posture == 1) a = new Notification("거북목 자세입니다!");
    else if (posture == 2) a = new Notification("고개/어깨 비대칭 자세입니다!");
  }, [posture]);

  const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: "user",
  };

  const WebcamCapture = () => {
    const webcamRef = useRef();

    const capture = () => {
      setTimeout(function () {
        const post = {
          p1: window.localStorage.getItem("p1"),
          p2: window.localStorage.getItem("p2"),
          p3: window.localStorage.getItem("p3"),
          p4: window.localStorage.getItem("p4"),
          p5: window.localStorage.getItem("p5"),
        };

        fetch("http://localhost:4001/pose", {
          method: "post",
          headers: {
            "content-type": "application/json",
          },
          body: JSON.stringify(post),
        })
          .then((res) => res.json())
          .then((json) => {
            setPosture(json.text);
          });
      }, 10);
    };

    useEffect(() => {
      setInterval(() => {
        if (webcamRef.current == null)
          imageSrc = "C:\\Users\\82109\\Downloads\\pose_capture.jpg";
        else imageSrc = webcamRef.current.getScreenshot();

        var a = document.createElement("a");
        a.style = "display: none";
        a.href = imageSrc;
        a.download = "pose_capture.jpg";

        document.body.appendChild(a);

        a.click();

        setTimeout(function () {
          // 다운로드가 안되는 경우 방지
          document.body.removeChild(a);
        }, 100);
        capture();
      }, 5000);
    }, []);

    return (
      <>
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          videoConstraints={videoConstraints}
          mirrored={true}
          height={0.6 * `${window.innerHeight}`}
          width={0.9 * `${window.innerWidth}`}
        />
      </>
    );
  };

  return (
    <div className="pose">
      <WebcamCapture />
      <div>
        {posture == 0 && "바른 자세입니다."}
        <div style={{ color: "red" }}>
          {posture == 1 && "거북목 자세입니다."}
          {posture == 2 && "고개/어깨 비대칭 자세입니다."}
        </div>
      </div>
    </div>
  );
}

export default Pose;
