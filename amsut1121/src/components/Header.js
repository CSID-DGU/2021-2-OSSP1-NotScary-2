import React from "react";
import { NavLink } from "react-router-dom";

function Header() {
  return (
    <div className="header">
      <NavLink
        className="h"
        to="/"
        onClick={() => {
          window.localStorage.clear();
        }}
      >
        AMSUT
      </NavLink>
    </div>
  );
}

export default Header;
