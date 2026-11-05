// SPDX-License-Identifier: MIT
// Test-only minimal ERC721. Local anvil use only.
pragma solidity ^0.8.24;

contract TestNFT {
    string public name;
    mapping(uint256 => address) public ownerOf;
    mapping(address => uint256) public balanceOf;
    mapping(uint256 => address) public getApproved;
    mapping(address => mapping(address => bool)) public isApprovedForAll;

    event Transfer(address indexed from, address indexed to, uint256 indexed id);
    event Approval(address indexed owner, address indexed spender, uint256 indexed id);
    event ApprovalForAll(address indexed owner, address indexed operator, bool approved);

    constructor(string memory n) { name = n; }

    function mint(address to, uint256 id) external { require(ownerOf[id] == address(0)); ownerOf[id] = to; balanceOf[to]++; emit Transfer(address(0), to, id); }

    function approve(address sp, uint256 id) external { require(ownerOf[id] == msg.sender, "owner"); getApproved[id] = sp; emit Approval(msg.sender, sp, id); }

    function setApprovalForAll(address op, bool ok) external { isApprovedForAll[msg.sender][op] = ok; emit ApprovalForAll(msg.sender, op, ok); }

    function transferFrom(address f, address to, uint256 id) public {
        require(ownerOf[id] == f, "from");
        require(msg.sender == f || getApproved[id] == msg.sender || isApprovedForAll[f][msg.sender], "auth");
        ownerOf[id] = to; balanceOf[f]--; balanceOf[to]++; delete getApproved[id];
        emit Transfer(f, to, id);
    }
}
