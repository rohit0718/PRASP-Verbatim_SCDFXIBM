import React from "react";
import {Button, Col, Container, Row} from "react-bootstrap";
import {BsCloudDownload} from 'react-icons/bs';
import {Link} from "react-router-dom";

const HistoryIndividualBar = ({name, serial}) =>{

    const start = "13:00";
    const randomStorage =  Math.floor( Math.random() * (115 - 100) + 100);
    const randomDuration = Math.floor( Math.random() * (45 - 30) + 30);

    const nameComp = (serial) ? (
        <div>
            { name }
        </div>
    ) : (
        <div style={{ fontSize: 'xx-large' }}>
            { name }
        </div>
    );

  return (
      <Container
          style = {{
              backgroundColor: "#F9F9FD",
              borderRadius: 10,
              marginTop: 15,
              padding: 15,
          }}

      >
          <Col   style={{
              textAlign : "start",
              marginLeft: 15
          }}>
              <Row

              >
                <Col>
                    <strong>
                        { nameComp }
                    </strong>
                    <i>{serial}</i>
                </Col>
                  <Col style={{
                      textAlign:"end",
                      marginTop: 15,
                      marginBottom: -15,

                  }}
                  >
                      <a href=  "SampleFootage.mp4"
                             download>
                      <Button
                          style={{
                              textAlign:"end",
                              marginRight: 15,

                          }}
                          variant="outline-primary"
                      >
                         <BsCloudDownload/> Raw Footage {randomStorage} mb
                      </Button>
                      </a>
                          <Link to=  "SampleReport.txt"
                                target="_blank" download>
                              <Button
                                  variant="outline-primary"
                              >
                              Download Report
                                </Button>
                          </Link>


                  </Col>

              </Row>
              <Row
              style = {{marginLeft:-10}}
              >
                  <Container>
                      Start: 1300 End: 13{randomDuration} Duration: {randomDuration} min
                  </Container>
              </Row>


          </Col>
      </Container>
  );

}
const HistoryIndividualBars = ({data})=>{
    let elements = [];
    console.log(data);
    for (let i  = 0;i <data.length;i++){
        elements.push(<HistoryIndividualBar
            serial={data[i]['serial']}
            name={data[i]['name']}
        />)
    }

    return (<div>
        {elements}
    </div>);
}

export default HistoryIndividualBars ;