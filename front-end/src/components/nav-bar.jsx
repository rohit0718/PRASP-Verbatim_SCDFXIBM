import React from "react";
import { Navbar, Nav, NavDropdown } from "react-bootstrap";
import { Link } from "react-router-dom";

const NavigationBar = () => {
    return  (
        <div className='topNavBar' id='top'>
            <Navbar collapseOnSelect expand="lg"  variant='dark'  sticky="top"  >
                <Nav className="me-auto" style ={{marginLeft:25}}>
                    <a className="navbar-brand" href="/">
                        <span style={{ fontSize: 30, fontWeight: 'bold' }}>Verbatim</span>
                    </a>
                </Nav>
                {/*<Navbar.Toggle aria-controls="responsive-navbar-nav"  style ={{marginRight:15}}/>*/}
                {/*<Navbar.Collapse id="responsive-navbar-nav">*/}
                {/*    <Nav className="ms-auto" style ={{marginRight:25}}>*/}
                {/*        <Link to='/history' className="nav-link">History</Link>*/}
                {/*        <Link to="/upload" className="nav-link">Upload</Link>*/}
                {/*    </Nav>*/}
                {/*</Navbar.Collapse>*/}
            </Navbar>
        </div>
    );
}

export default NavigationBar;
