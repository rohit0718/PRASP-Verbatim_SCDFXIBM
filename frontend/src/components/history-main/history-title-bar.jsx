import React from "react";
import {Row, Button, Container, Col} from "react-bootstrap";
import {BsCloudDownload} from 'react-icons/bs';
import {Link} from "react-router-dom";

const HistoryTitleBar = ()=>{

    const randomNum =  Math.floor( Math.random() * (550 - 500) + 500);
    return (
        <Container
            style = {{
                backgroundColor: "#F2F2FD",
                borderRadius: 10
            }}
        >
            <Row

            >
                <Col style={{
                    textAlign:"start",
                    marginTop: 15,
                    marginBottom: 15,
                    fontSize: 25
                }}>
                    <strong>
                        AI Generated Reports
                    </strong>
                </Col>

               <Col style={{
                   textAlign:"end",
                   marginTop: 15,

               }}>


                   <Link to={"AllFootage.zip"} target="_blank" download>
                       <Button
                           style={{
                               textAlign:"end",
                               marginRight: 15,

                           }}>
                           <BsCloudDownload/> All Footage {randomNum} mb
                       </Button>
                   </Link>




                  <Link to={"AllReports.zip"} target="_blank" download>
                      <Button >
                          Download All Reports
                      </Button>
                  </Link>
               </Col>


            </Row>
        </Container>
    );
}

export default HistoryTitleBar;