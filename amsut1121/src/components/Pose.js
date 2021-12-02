import React, { useState, useEffect } from "react";
import { Component } from "react/cjs/react.production.min";
import * as tf from "@tensorflow/tfjs";

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
  //서버로 5가지 좌표값을 통해 분석된 자세 받아오는 함수

/*
    fetch("http://localhost:4000/posture", {
      // /posture를 post를 통해 서버와 연동
      method: "get",
      headers: {
        "content-type": "application/json",
      },
    })
      .then((res) => res.json())
      .then((json) => {
          setPosture(posture+1);
      });
    */
    
    
        
    useEffect(() => {
        let a;
        if(posture==1) a = new Notification("거북목 자세입니다!");
        else if(posture == 2) a = new Notification("턱을 괸 자세입니다!");
        else if(posture == 3) a = new Notification("양쪽 어깨와 눈이 모두 나오도록 위치해주세요!")
    }, [posture])
    
    
    
    return (
      <div className="pose">
            {posture == 0 && "바른 자세입니다."}
            <button onClick={() => setPosture("0")}>바</button>
            <button onClick={() => setPosture("1")}>거</button>
            <button onClick={() => setPosture("2")}>턱</button>
            <button onClick={() => setPosture("3")}>오</button>
            
        <div style={{ color: "red" }}>
            {posture == 1 && "거북목 자세입니다."}
            {posture == 2 && "턱을 괸 자세입니다."}
            {posture == 3 && "양쪽 어깨와 눈이 모두 나오도록 위치해주세요."}
        </div>
      </div>
    );
  
}

export default Pose;