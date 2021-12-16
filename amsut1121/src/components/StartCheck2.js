import React, { useState, useEffect, useRef } from "react";
import { Redirect } from "react-router-dom";
import Webcam from "react-webcam";

function StartCheck2() {
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
      <>
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          videoConstraints={videoConstraints}
          mirrored={true}
          height={0.6 * `${window.innerHeight}`}
          width={0.9 * `${window.innerWidth}`}
          style={{ marginTop: "10vh" }}
        />
      </>
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
          "양쪽 어깨와 눈이 보이도록 설정 후 5초 동안 자세를 유지해주세요."}

        <div style={{ color: "red" }}>
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
      {posture == 1 && <Redirect to={{ pathname: "/pose" }} />}
    </div>
  );
}

export default StartCheck2;
