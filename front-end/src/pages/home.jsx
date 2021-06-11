import React from "react";
import MainMenuBox from "../components/main-menu-box";
import  {Row, Container,Col} from "react-bootstrap";

const Home = ()=>{
    return (
        <Container>
            <Row>
                <Col style={{}} >
                    <MainMenuBox name={"History"}/>
                </Col>
                <Col >
                    <MainMenuBox name={"Upload"}/>
                </Col>
            </ Row>
        </Container>
    );
}

export default Home;