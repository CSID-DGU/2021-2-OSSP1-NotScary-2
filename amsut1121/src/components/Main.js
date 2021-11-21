import React from "react";
import Webcam from "react-webcam";

const videoConstraints = {
  width: 1280,
  height: 720,
  facingMode: "user",
};

function Main() {
  const webcamRef = React.useRef(null);

  return (
    <>
      <div className="main">
        양쪽 어깨와 눈이 보이도록 설정 후 5초 동안 자세를 유지해주세요.
        <br />
        <br />
        <Webcam
          audio={false}
          ref={webcamRef}
          mirrored={true}
          height={0.6 * `${window.innerHeight}`}
          width={0.9 * `${window.innerWidth}`}
          videoConstraints={videoConstraints}
        />
        <br />
        <br />
        <div style={{ color: "red" }}>어깨 좌표값이 인식되지 않았습니다!</div>
      </div>
    </>
  );
}

export default Main;
