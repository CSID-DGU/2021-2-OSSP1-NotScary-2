import React, { useState, useEffect, useRef } from "react";
import Webcam from "react-webcam";
import moment from "moment";
import "./poseHistory.css"

function PoseHistory(pros) {
    const [getMoment, setMoment] = useState(moment());

    const today = getMoment;
    const firstWeek = today.clone().startOf('month').week();
    const lastWeek = today.clone().endOf('month').week() == 1 ? 53: today.clone().endOf('month').week();

    const calendarArr = () => {
        let result = [];
        let week = firstWeek;
        for(week; week<=lastWeek; week++) {
            result = result.concat(
                <tr key = {week} className = "defaultTr">
                    {
                        Array(7).fill(0).map((data, index) => {
                            let days = today.clone().startOf('year').week(week).startOf('week').add(index, 'day');
                            if(moment().format('YYYYMMDD') === days.format('YYYYMMDD')) {
                                return (
                                    <td key={index}  className="todayTd">
                                        <span>{days.format('D')}</span>
                                    </td>
                                );
                            } else if(days.format('MM') !== today.format('MM')) {
                                return (
                                    <td key={index} className="notThisMonthTd">
                                        <span>{days.format('D')}</span>
                                    </td>
                                );
                            } else {
                                return (
                                    <td key={index} className="defaultTd">
                                        <span>{days.format('D')}</span>
                                    </td>
                                )
                            }
                        })
                    }
                </tr>
            )
        }
        return result;
    }


    return (
        <div>
            <div className = "control">
                <button onClick = {() => {setMoment(getMoment.clone().subtract(1, 'month'))}}>이전 달</button>
                <span>{today.format('YYYY년 MM월')}</span>
                <button onClick = {() => {setMoment(getMoment.clone().add(1, 'month'))}}>다음 달</button>
            </div>
            <table>
                <tr>
                    <td style={{color: "red"}}>일</td> 
                    <td>월</td>
                    <td>화</td>
                    <td>수</td>
                    <td>목</td>
                    <td>금</td>
                    <td style={{color: "blue"}}>토</td>
                </tr>
                <tbody>
                    {calendarArr()}
                </tbody>
            </table>
        </div>
    );
}

export default PoseHistory;
