import React from "react";
import { Button } from "react-bootstrap";
import { Link } from "react-router-dom";
const MainMenuBox = ({name})=>{
    const page = "/" + name;
    return(<div >
            <Link to= {page}>
                <Button style={{
                    height: 250,
                    width: 250,
                    margin: 50,
                    fontSize: 30,
                    borderRadius: 25
                }}> {name}
                </Button>
            </Link>
    </div>);
}

export default MainMenuBox;