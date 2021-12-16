import React, { useState, useEffect, useRef } from "react";
import { Redirect } from "react-router-dom";
import Webcam from "react-webcam";
import goldman from "./goldman.png";

function StartCheck2() {
  var poseArr = [];

  const [posture, setPosture] = useState(0);

  const SC = () => {
    const webcamRef = useRef();

    const capture = () => {
      setTimeout(function () {
        var imageSrc;

        console.log(webcamRef.current);

        if (webcamRef.current == null)
          imageSrc = "C:\\Users\\82109\\Downloads\\start_capture.jpg";
        else imageSrc = webcamRef.current.getScreenshot();

        console.log("asdf");
        fetch("http://localhost:4000/startcheck", {
          // /posture를 post를 통해 서버와 연동
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
      }, 10);

      console.log(poseArr);

      localStorage.setItem("p1", poseArr[1]);
      localStorage.setItem("p2", poseArr[2]);
      localStorage.setItem("p3", poseArr[3]);
      localStorage.setItem("p4", poseArr[4]);
      localStorage.setItem("p5", poseArr[5]);
      localStorage.setItem("p6", poseArr[6]);
      localStorage.setItem("p7", poseArr[7]);

      localStorage.getItem("p1");
      localStorage.getItem("p2");
      localStorage.getItem("p3");
    };

    useEffect(() => {
      setInterval(() => {
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
              top: "3vw",
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
        {posture}
        {posture == 0 &&
          "양쪽 어깨와 눈이 보이도록 설정 후 10초 동안 바른 자세를 유지해주세요."}

        <div style={{ color: "red" }}>
          {poseArr[0] == -1 && "바른 자세를 유지해주세요."}

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
      {console.log(localStorage.getItem("p1"))}
      {localStorage.getItem("p1") == "1" && (
        <Redirect to={{ pathname: "/pose" }} />
      )}
      {localStorage.getItem("p1") == "-1" && (
        <Redirect to={{ pathname: "/pose" }} />
      )}
    </div>
  );
}

export default StartCheck2;
