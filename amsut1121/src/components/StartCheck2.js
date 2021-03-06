import React, { useState, useEffect, useRef } from "react";
import { Redirect } from "react-router-dom";
import Webcam from "react-webcam";
import goldman from "./goldman.png";

var imageSrc;
var count = 1;

function StartCheck2() {
  var poseArr = [];
  var interval;

  const [posture, setPosture] = useState(0);

  const SC = () => {
    const webcamRef = useRef();
    const capture = () => {
      setTimeout(function () {
        fetch("http://localhost:4000/startcheck", {
          method: "post",
          headers: {
            "content-type": "application/json",
          },
        })
          .then((res) => res.json())
          .then((json) => {
            setPosture(json.text);
            poseArr = json.text.split("#");
          });
      }, 10);

      if (poseArr[0] == "-1") setPosture(-1);

      if (poseArr[0] == "1" && count) {
        console.log("좋은 자세");
        window.localStorage.setItem("start", poseArr[0]);
        window.localStorage.setItem("p1", poseArr[1]);
        window.localStorage.setItem("p2", poseArr[2]);
        window.localStorage.setItem("p3", poseArr[3]);
        window.localStorage.setItem("p4", poseArr[4]);
        window.localStorage.setItem("p5", poseArr[5]);

        clearInterval(interval);
        count = 0;
      }
    };

    useEffect(() => {
      interval = setInterval(() => {
        if (webcamRef.current == null)
          imageSrc = "C:\\Users\\82109\\Downloads\\start_capture.jpg";
        else imageSrc = webcamRef.current.getScreenshot();

        var a = document.createElement("a");
        a.style = "display: none";
        a.href = imageSrc;
        a.download = "start_capture.jpg";

        document.body.appendChild(a);

        a.click();

        setTimeout(function () {
          // 다운로드가 안되는 경우 방지
          document.body.removeChild(a);
        }, 100);

        capture();
      }, 5000);
    }, []);

    const videoConstraints = {
      width: 1280,
      height: 720,
      facingMode: "user",
    };

    return (
      <div style={{ position: "relative", textAlign: "center" }}>
        <div style={{ textAlign: "center" }}>
          <img
            src={goldman}
            alt=""
            style={{
              position: "absolute",
              width: "20vw",
              zIndex: "3",
              left: "39vw",
              top: "6vh",
              opacity: "0.7",
            }}
          />
        </div>

        <div
          style={{
            marginTop: "10vh",
          }}
        >
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            videoConstraints={videoConstraints}
            mirrored={true}
            height={0.6 * `${window.innerHeight}`}
            width={0.9 * `${window.innerWidth}`}
          />
        </div>
      </div>
    );
  };

  return (
    <div className="start">
      <SC />
      <div className="main2">
        <br />
        <div className="loader10" />
        <br />
        {posture == 0 &&
          "양쪽 어깨와 눈이 보이도록 설정 후 10초 동안 바른 자세를 유지해주세요."}

        <div style={{ color: "red" }}>
          {posture == -1 && "바른 자세를 유지해주세요."}
          {posture == 2 && "오른쪽 어깨가 확인되지 않았습니다."}
          {posture == 5 && "왼쪽 어깨가 확인되지 않았습니다."}
          {posture == 14 && "오른쪽 눈이 확인되지 않았습니다."}
          {posture == 15 && "왼쪽 눈이 확인되지 않았습니다."}
          {posture == 7 && "오른쪽 어깨, 왼쪽 어깨가 확인되지 않았습니다."}
          {posture == 16 && "오른쪽 어깨, 오른쪽 눈이 확인되지 않았습니다."}
          {posture == 17 && "오른쪽 어깨, 왼쪽 눈이 확인되지 않았습니다."}
          {posture == 19 && "왼쪽 어깨, 오른쪽 눈이 확인되지 않았습니다."}
          {posture == 20 && "왼쪽 어깨, 왼쪽 눈이 확인되지 않았습니다."}
          {posture == 29 && "오른쪽 눈, 왼쪽 눈이 확인되지 않았습니다."}
          {posture == 21 &&
            "오른쪽 어깨, 왼쪽 어깨, 오른쪽 눈이 확인되지 않았습니다."}
          {posture == 22 &&
            "오른쪽 어깨, 왼쪽 어깨, 왼쪽 눈이 확인되지 않았습니다."}
          {posture == 31 &&
            "오른쪽 어깨, 오른쪽 눈, 왼쪽 눈이 확인되지 않았습니다."}
          {posture == 34 &&
            "왼쪽 어깨, 오른쪽 눈, 왼쪽 눈이 확인되지 않았습니다."}
          {posture == 36 &&
            "오른쪽 어깨, 왼쪽 어깨, 오른쪽 눈, 왼쪽 눈이 확인되지 않았습니다."}
        </div>
      </div>
      {/* 양쪽 눈과 어깨가 모두 파악된 바른 자세인 경우 자세 판별 페이지로 이동 */}
      {window.localStorage.getItem("start") == "1" && (
        <Redirect to={{ pathname: "/pose" }} />
      )}
    </div>
  );
}

export default StartCheck2;
