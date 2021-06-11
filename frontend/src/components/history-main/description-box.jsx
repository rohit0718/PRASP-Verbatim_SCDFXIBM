import React, {useEffect, useState} from "react";
import {Row, Col, Container} from 'react-bootstrap';

const HistoryDescriptionBox = ({selectedData})=>{
    const location = selectedData['location'];
    const time = selectedData['date'];
    const description = selectedData['description'];
    const men = selectedData["men"];
    let menString = "";
    for (let i = 1 ; i < men.length - 1 ; i ++ ){
        menString = menString + men[i] + ", ";
    }
    menString += " and " + men[men.length - 1];


    return (
        <Container
            style={{
                marginBottom:17,
                backgroundColor: "#F2F2FD",
                paddingTop: 15,
                paddingBottom: 15,
                borderRadius: 10
            }}
        >
            <Col style={{
                textAlign: "start"
            }}>
                <Row>
                   <div>
                       <strong>Time:</strong> {time}
                   </div>
                </Row>
                <Row>
                      <div>
                          <strong>Location:</strong> {location}</div>
                </Row>
                <Row>
                      <div>
                          <strong> Description:    </strong> {description}
                      </div>
                </Row>
                <Row>
                    <div>
                        <strong> Unit IC:    </strong> {men[0]}
                    </div>
                </Row>
                <Row>
                    <div>
                        <strong> Men on duty:    </strong> {menString}
                    </div>
                </Row>
            </Col>
        </Container>
    );
}

export default HistoryDescriptionBox;