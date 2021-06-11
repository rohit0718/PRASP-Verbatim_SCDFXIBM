import React from "react";
import IndividualIncidentButtons from "./indiv-side-bar";

const HistorySideBar = ({sampleData, setSelectedIndex, currentlySelected})=>{



    let elements = [];
    for (let i = 0; i < sampleData.length;i++){

        elements.push(<IndividualIncidentButtons
            date={sampleData[i]['date'].toString()}
            location={sampleData[i]['location'].toString()}
            index = {i}
            setSelectedIndex = {setSelectedIndex}
            currentlySelected={currentlySelected}
        />)
    }
    return (
        <div style={{
            marginLeft: 45,
            marginRight: 30,
        }}>
        {elements}
        </div>
    );

}

export default HistorySideBar ;