// 자세 판별 서버(Pose.js와 연결)

const express = require("express");
const app = express();
const port = 4001;
const cors = require("cors");
const bodyParser = require("body-parser");

app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());
app.use(cors());

app.post("/pose", (req, res) => {
  const p1 = req.body.p1;
  const p2 = req.body.p2;
  const p3 = req.body.p3;
  const p4 = req.body.p4;
  const p5 = req.body.p5;

  const { spawn } = require("child_process");
  const result_01 = spawn("python", ["./webcam_model4.py", p1, p2, p3, p4, p5]);
  const sendText = { text: "" };

  result_01.stdout.on("data", (result) => {
    console.log(result.toString());
    sendText.text = result.toString();
    res.send(sendText);
  });

  setTimeout(function () {}, 2000);

  console.log("삭제 시작");
  var fs = require("fs"); // 다운로드된 파일 경로 (개인 pc 경로에 맞게 수정해줘야됨)
  fs.unlink(`C:\\Users\\82109\\Downloads\\pose_capture.jpg`, (err) => {
    if (err) console.log(err);
    else {
      console.log("삭제 끝");
    }
  });
});

app.listen(port, () => {
  console.log(`Connect at http://localhost:${port}`);
});
