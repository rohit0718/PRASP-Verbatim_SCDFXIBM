import React from "react";
import {Button, Row} from "react-bootstrap";

const IndividualIncidentButtons = ({date, location, setSelectedIndex, index, currentlySelected})=>{
    const variantStyle = currentlySelected === index ? "primary": "secondary";
    return(
        <Button
            variant= {variantStyle}
            style={{
                marginTop: 25,
                paddingTop: 5,
                paddingBottom: 5,
                paddingLeft: 50,
                paddingRight: 50,
            }}
            onClick={()=>{
                setSelectedIndex(index);
            }}
        >
            <div>
                <div>
                    {date }
                </div>
                <div>
                    { location}
                </div>
            </div>


        </Button>
    );
}

export default IndividualIncidentButtons;