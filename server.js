const express = require("express"); 
const app = express();
const port = 4000; 
const cors = require("cors");
const bodyParser = require("body-parser");

app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());
app.use(cors());


app.post('/posture', (req, res) => {
    const SD=req.body.SD
    const ED=req.body.ED
    const SED=req.body.SED
    const SS=req.body.SS
    const ES=req.body.ES

    const { spawn } = require('child_process');
    const result_02 = spawn('python', ['modelExample.py', SD,ED,SED,SS,ES]);
    const sendText = {
        text : "",
    };
    console.log("시작");
    result_02.stdout.on('data', function(data){
        console.log(data.toString());
        sendText.text=data.toString();
        res.send(sendText);
    });
    
})

app.listen(port, ()=>{
    console.log(`Connect at http://localhost:${port}`);
})