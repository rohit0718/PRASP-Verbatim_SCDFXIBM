import React,  { useState } from "react";
import  {Row, Container,Col} from "react-bootstrap";
import HistorySideBar from "../components/history-side-bar/side-bar";
import HistoryDescriptionBox from "../components/history-main/description-box";
import HistoryTitleBar from "../components/history-main/history-title-bar";
import HistoryIndividualBars from "../components/history-main/indiv-bar";
import sampleData from "../sampleData.json";


const tempStyle = {
    // borderStyle : "solid",
    padding : 0,
    alignContent: "flex-start",
    alignItems: "flex-start",
    justifyContent:"start",
    marginRight: 15
}

const HistoryPage = () => {
    const [selectedIndex, setSelectedIndex] = useState(0);

    return (
        <Container fluid
           >
            <Row>
                <Col md={3}  style={tempStyle}>
                    <HistorySideBar
                        setSelectedIndex={setSelectedIndex}
                        sampleData={sampleData}
                        currentlySelected={selectedIndex}
                    />
                </Col>
                <Col md={8} style={tempStyle}>
                    <HistoryDescriptionBox
                        selectedData = {sampleData[selectedIndex]}
                    />
                    <HistoryTitleBar/>
                    <HistoryIndividualBars
                        data={sampleData[selectedIndex]["reports"]}
                    />
                </Col>

            </Row>
        </Container>
    );
}

export default HistoryPage;