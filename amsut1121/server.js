const express = require("express");
const app = express();
const port = 4000;
const cors = require("cors");
const bodyParser = require("body-parser");

app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());
app.use(bodyParser.json({ limit: "50mb" }));
app.use(
  bodyParser.urlencoded({
    limit: "50mb",
    extended: true,
    parameterLimit: "100000",
  })
);
app.use(cors());

app.post("/test", (req, res) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.set({ "access-control-allow-origin": "*" });

  const { spawn } = require("child_process");
  const result_01 = spawn("python", ["./test.py"]);
  const sendText = { text: "" };
  console.log("imagetest 시작");

  result_01.stdout.on("data", (result) => {
    console.log(result.toString());
    sendText.text = result.toString();
    res.send(sendText);
    console.log("imagetest 끝");
  });

  console.log("삭제시작");

  var fs = require("fs"); // 다운로드된 파일 경로 (개인 pc 경로에 맞게 수정해줘야됨)
  fs.unlink(`C:\\Users\\82109\\Downloads\\pose_capture.jpg`, (err) => {
    if (err) console.log(err);
    else {
      console.log("삭제끝");
    }
  });
});

app.post("/startcheck", (req, res) => {
  const { spawn } = require("child_process");
  const result_01 = spawn("python", ["./start_check.py"]); //5가지 좌표값을 통해 자세 판별 모델 파이썬 파일을 이용

  const sendText = { text: "" }; //분석된 자세를 다시 전송할
  console.log("시작11");

  result_01.stdout.on("data", (result) => {
    console.log(result.toString());
    sendText.text = result.toString();
    res.send(sendText);
  });

  console.log("끝11");
});

app.post("/posture", (req, res) => {
  //분석하고자하는 자세의 좌표값
  //ShoulderDistance,EyeDistance,ShoulderEyeDistance,ShoulderSlope,EyeSlope
  // const SD = req.body.SD;
  // const ED = req.body.ED;
  // const SED = req.body.SED;
  // const SS = req.body.SS;
  // const ES = req.body.ES;

  const { spawn } = require("child_process");
  const result_02 = spawn("python", ["./webcam_model2.py"]);
  // const result_02 = spawn("python", ["./modelExample.py", SD, ED, SED, SS, ES]); //5가지 좌표값을 통해 자세 판별 모델 파이썬 파일을 이용
  const sendText = { text: "" }; //분석된 자세를 다시 전송할
  console.log("시작");
  result_02.stdout.on("data", (result) => {
    console.log(result.toString());
    sendText.text = result.toString();
    res.send(sendText);
  });
  console.log("끝");
});

app.listen(port, () => {
  console.log(`Connect at http://localhost:${port}`);
});
